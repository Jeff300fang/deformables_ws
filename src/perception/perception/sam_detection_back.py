#!/usr/bin/env python3

import sys
from pathlib import Path
from contextlib import nullcontext

import cv2 as cv
import numpy as np
import torch
import rclpy

from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


# Allow local sam3 package discovery
for workspace_root in [Path.cwd(), *Path(__file__).resolve().parents]:
    for src_dir in (workspace_root, workspace_root / "src"):
        local_package_path = src_dir / "sam3"
        if local_package_path.exists() and str(local_package_path) not in sys.path:
            sys.path.insert(0, str(local_package_path))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SingleCameraSAMRopeNode(Node):
    def __init__(self):
        super().__init__("single_camera_sam_rope_node")

        self.declare_parameter(
            "sam_checkpoint_path",
            "/home/jeff/trustworthroboticsgroup/CoRL2026/deformables_ws/src/perception/checkpoints/sam3.pt",
        )
        self.declare_parameter("sam_prompt", "rope")
        self.declare_parameter("sam_confidence_threshold", 0.35)
        self.declare_parameter("run_sam_every_frame", True)
        self.declare_parameter("resize_width", 160)

        self.sam_checkpoint_path = str(self.get_parameter("sam_checkpoint_path").value)
        self.sam_prompt = str(self.get_parameter("sam_prompt").value)
        self.sam_confidence_threshold = float(
            self.get_parameter("sam_confidence_threshold").value
        )
        self.run_sam_every_frame = bool(
            self.get_parameter("run_sam_every_frame").value
        )
        self.resize_width = int(self.get_parameter("resize_width").value)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.sam_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.bridge = CvBridge()

        self.get_logger().info(f"Using device: {self.device}")
        self.get_logger().info("Loading SAM image model...")

        sam_model = build_sam3_image_model(
            checkpoint_path=self.sam_checkpoint_path,
            device=str(self.device),
        )
        sam_model = sam_model.eval()

        self.sam_processor = Sam3Processor(
            sam_model,
            device=str(self.device),
            confidence_threshold=self.sam_confidence_threshold,
        )

        self.last_mask = None

        self.image_sub = self.create_subscription(
            Image,
            "/back_camera/color/image_raw",
            self.image_callback,
            10,
        )

        self.mask_pub = self.create_publisher(
            Image,
            "/back_camera/sam_rope/mask",
            10,
        )

        self.annotated_pub = self.create_publisher(
            CompressedImage,
            "/back_camera/sam_rope/annotated_image/compressed",
            10,
        )

        self.get_logger().info("Publishing mask to /sam_rope/mask")
        self.get_logger().info(
            "Publishing annotated image to /sam_rope/annotated_image/compressed"
        )

    def maybe_resize(self, rgb):
        if self.resize_width <= 0:
            return rgb, 1.0

        h, w = rgb.shape[:2]
        scale = self.resize_width / float(w)
        new_h = int(round(h * scale))

        resized = cv.resize(
            rgb,
            (self.resize_width, new_h),
            interpolation=cv.INTER_AREA,
        )

        return resized, scale

    def run_sam(self, rgb):
        image = PILImage.fromarray(rgb)

        if self.device.type == "cuda":
            autocast_context = torch.autocast(
                device_type="cuda",
                dtype=self.sam_dtype,
            )
        else:
            autocast_context = nullcontext()

        with torch.inference_mode(), autocast_context:
            sam_state = self.sam_processor.set_image(image)
            sam_state = self.sam_processor.set_text_prompt(
                prompt=self.sam_prompt,
                state=sam_state,
            )

        masks = sam_state.get("masks")
        scores = sam_state.get("scores")

        if masks is None or len(masks) == 0:
            return None

        if torch.is_tensor(masks):
            masks_np = masks.detach().to(dtype=torch.float32).cpu().numpy()
        else:
            masks_np = np.asarray(masks)

        if torch.is_tensor(scores):
            scores_np = scores.detach().to(dtype=torch.float32).cpu().numpy()
        elif scores is None:
            scores_np = None
        else:
            scores_np = np.asarray(scores)

        if masks_np.shape[0] == 0:
            return None

        if scores_np is not None and scores_np.shape[0] == masks_np.shape[0]:
            best_idx = int(np.argmax(scores_np))
        else:
            areas = np.sum(masks_np.reshape(masks_np.shape[0], -1) > 0, axis=1)
            best_idx = int(np.argmax(areas))

        mask = np.squeeze(masks_np[best_idx])
        binary_mask = mask > 0.0

        if np.sum(binary_mask) < 20:
            return None

        return binary_mask.astype(np.uint8)

    def draw_mask_overlay(self, rgb, mask):
        vis = rgb.copy()

        overlay = np.zeros_like(vis)
        overlay[:, :, 1] = mask * 255

        vis = cv.addWeighted(vis, 0.75, overlay, 0.35, 0.0)

        contours, _ = cv.findContours(
            mask,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE,
        )

        cv.drawContours(vis, contours, -1, (255, 0, 0), 2)

        cv.putText(
            vis,
            f"SAM prompt: {self.sam_prompt}",
            (20, 35),
            cv.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        return vis

    def publish_annotated(self, rgb, header):
        bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

        ok, encoded = cv.imencode(
            ".jpg",
            bgr,
            [int(cv.IMWRITE_JPEG_QUALITY), 90],
        )

        if not ok:
            return

        msg = CompressedImage()
        msg.header = header
        msg.format = "jpeg"
        msg.data = encoded.tobytes()

        self.annotated_pub.publish(msg)

    def image_callback(self, msg):
        # TODO: Enter gripper condition to determine which camera to run
        return
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)

            input_rgb, scale = self.maybe_resize(frame_rgb)

            if self.run_sam_every_frame or self.last_mask is None:
                mask = self.run_sam(input_rgb)

                if mask is None:
                    self.get_logger().warn("SAM found no rope mask")
                    return

                self.last_mask = mask
            else:
                mask = self.last_mask

            if scale != 1.0:
                h, w = frame_rgb.shape[:2]
                mask_full = cv.resize(
                    mask,
                    (w, h),
                    interpolation=cv.INTER_NEAREST,
                )
            else:
                mask_full = mask

            mask_msg = self.bridge.cv2_to_imgmsg(
                (mask_full * 255).astype(np.uint8),
                encoding="mono8",
            )
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)

            annotated_rgb = self.draw_mask_overlay(frame_rgb, mask_full)
            self.publish_annotated(annotated_rgb, msg.header)

            self.get_logger().info(
                f"Published rope mask. pixels={int(np.sum(mask_full > 0))}"
            )

        except Exception as e:
            self.get_logger().error(f"Image callback failed: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = SingleCameraSAMRopeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()