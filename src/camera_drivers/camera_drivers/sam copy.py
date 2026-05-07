#!/usr/bin/env python3

import sys
import time
import uuid
import shutil
import tempfile
import traceback
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage

for workspace_root in [Path.cwd(), *Path(__file__).resolve().parents]:
    for src_dir in (workspace_root, workspace_root / "src"):
        sam_path = src_dir / "sam3"
        if sam_path.exists() and str(sam_path) not in sys.path:
            sys.path.insert(0, str(sam_path))

from sam3.model_builder import build_sam3_multiplex_video_predictor


class SAM31VideoNode(Node):
    def __init__(self):
        super().__init__("sam31_video_node")

        self.declare_parameter("image_topic", "/front_camera/color/image_raw")
        self.declare_parameter("annotated_topic", "/sam31/annotated/compressed")
        self.declare_parameter("mask_topic", "/sam31/mask")
        self.declare_parameter(
            "checkpoint_path",
            "/home/jeffreyfang/deformables/src/sam3/checkpoints/sam3.1_multiplex.pt",
        )
        self.declare_parameter("prompt", "rope")
        self.declare_parameter("confidence_threshold", 0.15)
        self.declare_parameter("publish_mask", True)
        self.declare_parameter("process_every_n_frames", 1)
        self.declare_parameter("clip_length", 3)
        self.declare_parameter("loop", True)
        self.declare_parameter("offload_video_to_cpu", False)
        self.declare_parameter("resize_width", 384)
        self.declare_parameter("resize_height", 216)
        self.declare_parameter("use_shm_tmp", True)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.annotated_topic = str(self.get_parameter("annotated_topic").value)
        self.mask_topic = str(self.get_parameter("mask_topic").value)
        self.checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        self.prompt = str(self.get_parameter("prompt").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.publish_mask = bool(self.get_parameter("publish_mask").value)
        self.process_every_n_frames = int(self.get_parameter("process_every_n_frames").value)
        self.clip_length = int(self.get_parameter("clip_length").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.offload_video_to_cpu = bool(self.get_parameter("offload_video_to_cpu").value)
        self.resize_width = int(self.get_parameter("resize_width").value)
        self.resize_height = int(self.get_parameter("resize_height").value)
        self.use_shm_tmp = bool(self.get_parameter("use_shm_tmp").value)

        self.bridge = CvBridge()
        self.frame_count = 0
        self.processing = False
        self.frames = []
        self.headers = []

        if not torch.cuda.is_available():
            raise RuntimeError("SAM 3.1 multiplex video predictor requires CUDA.")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.get_logger().info("Loading SAM 3.1 multiplex video model...")

        self.predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=self.checkpoint_path,
            default_output_prob_thresh=self.confidence_threshold,
            async_loading_frames=False,
            use_fa3=False,
            use_rope_real=False,
            compile=False,
            warm_up=False,
            max_num_objects=16,
            multiplex_count=16,
        )

        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.annotated_pub = self.create_publisher(
            CompressedImage,
            self.annotated_topic,
            10,
        )

        self.mask_pub = self.create_publisher(
            Image,
            self.mask_topic,
            10,
        )

        self.get_logger().info(f"Subscribed to: {self.image_topic}")
        self.get_logger().info(f"Publishing annotated image to: {self.annotated_topic}")
        self.get_logger().info(f"Publishing mask image to: {self.mask_topic}")
        self.get_logger().info(f"Prompt: {self.prompt}")
        self.get_logger().info(f"Clip length: {self.clip_length}")
        self.get_logger().info(f"Resize: {self.resize_width}x{self.resize_height}")

    def image_callback(self, msg: Image):
        if self.processing:
            return

        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return

        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            if self.resize_width > 0 and self.resize_height > 0:
                frame_bgr = cv.resize(
                    frame_bgr,
                    (self.resize_width, self.resize_height),
                    interpolation=cv.INTER_AREA,
                )

            self.frames.append(frame_bgr.copy())
            self.headers.append(msg.header)

            if len(self.frames) >= self.clip_length:
                self.processing = True
                self.process_clip()

        except Exception as e:
            self.processing = False
            self.get_logger().error(f"SAM 3.1 callback failed: {e}")
            self.get_logger().error(traceback.format_exc())

    def process_clip(self):
        tmp_root = "/dev/shm" if self.use_shm_tmp and Path("/dev/shm").exists() else None
        clip_dir = Path(tempfile.mkdtemp(prefix="sam31_clip_", dir=tmp_root))
        session_id = None

        try:
            frames = self.frames
            headers = self.headers
            self.frames = []
            self.headers = []

            for idx, frame_bgr in enumerate(frames):
                cv.imwrite(str(clip_dir / f"{idx:05d}.jpg"), frame_bgr)

            session_id = self.start_video_session(str(clip_dir))

            with torch.inference_mode():
                response = self.predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=0,
                        text=self.prompt,
                        output_prob_thresh=self.confidence_threshold,
                    )
                )

                outputs_by_frame = {
                    response["frame_index"]: response["outputs"]
                }

                for response in self.predictor.handle_stream_request(
                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                        output_prob_thresh=self.confidence_threshold,
                    )
                ):
                    outputs_by_frame[response["frame_index"]] = response["outputs"]

            latest_idx = len(frames) - 1
            frame_bgr = frames[latest_idx]
            header = headers[latest_idx]

            outputs = outputs_by_frame.get(latest_idx, {})
            masks = self.extract_masks(outputs, frame_bgr.shape[:2])
            binary_mask = self.combine_masks(masks, frame_bgr.shape[:2])
            annotated_bgr = self.draw_masks(frame_bgr, masks)

            self.publish_annotated(annotated_bgr, header)

            if self.publish_mask:
                mask_msg = self.bridge.cv2_to_imgmsg(binary_mask, encoding="mono8")
                mask_msg.header = header
                self.mask_pub.publish(mask_msg)

            self.get_logger().info(
                f"Published latest SAM result from {len(frames)}-frame clip"
            )

        except Exception as e:
            self.get_logger().error(f"SAM 3.1 video inference failed: {e}")
            self.get_logger().error(traceback.format_exc())

        finally:
            if session_id is not None:
                try:
                    self.predictor.handle_request(
                        request=dict(
                            type="close_session",
                            session_id=session_id,
                        )
                    )
                except Exception:
                    pass

            shutil.rmtree(clip_dir, ignore_errors=True)
            self.processing = False

            if not self.loop:
                self.destroy_subscription(self.sub)

            if torch.cuda.is_available:
                torch.cuda.empty_cache()

    def start_video_session(self, resource_path):
        inference_state = self.predictor.model.init_state(
            resource_path=resource_path,
            offload_video_to_cpu=self.offload_video_to_cpu,
            async_loading_frames=False,
        )

        session_id = str(uuid.uuid4())

        self.predictor._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }

        return session_id

    def extract_masks(self, outputs, frame_shape):
        masks = outputs.get("out_binary_masks")

        if masks is None:
            return np.zeros((0, frame_shape[0], frame_shape[1]), dtype=bool)

        if torch.is_tensor(masks):
            masks = masks.detach().cpu().numpy()
        else:
            masks = np.asarray(masks)

        masks = np.squeeze(masks)

        if masks.ndim == 2:
            masks = masks[None]

        if masks.size == 0:
            return np.zeros((0, frame_shape[0], frame_shape[1]), dtype=bool)

        return masks.astype(bool)

    def combine_masks(self, masks, frame_shape):
        if masks.shape[0] == 0:
            return np.zeros(frame_shape, dtype=np.uint8)

        return np.any(masks, axis=0).astype(np.uint8) * 255

    def draw_masks(self, frame_bgr, masks):
        overlay = frame_bgr.copy()

        if masks.shape[0] == 0:
            return overlay

        colors = [
            np.array([0, 0, 255], dtype=np.uint8),
            np.array([0, 255, 255], dtype=np.uint8),
            np.array([255, 0, 0], dtype=np.uint8),
            np.array([0, 255, 0], dtype=np.uint8),
            np.array([255, 0, 255], dtype=np.uint8),
        ]

        for idx, mask in enumerate(masks):
            color = colors[idx % len(colors)]

            overlay[mask] = overlay[mask] * 0.65 + color * 0.35

            binary_mask = mask.astype(np.uint8) * 255
            contours, _ = cv.findContours(
                binary_mask,
                cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE,
            )
            cv.drawContours(overlay, contours, -1, color.tolist(), 2)

        return overlay

    def publish_annotated(self, bgr_img, header):
        ok, encoded = cv.imencode(
            ".jpg",
            bgr_img,
            [int(cv.IMWRITE_JPEG_QUALITY), 85],
        )

        if not ok:
            return

        msg = CompressedImage()
        msg.header = header
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self.annotated_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SAM31VideoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()