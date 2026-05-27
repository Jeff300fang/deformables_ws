#!/usr/bin/env python3

import sys
from pathlib import Path
from contextlib import nullcontext
from collections import deque

import cv2 as cv
import numpy as np
import torch
import rclpy

from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
from skimage.morphology import skeletonize
from std_msgs.msg import Bool


for workspace_root in [Path.cwd(), *Path(__file__).resolve().parents]:
    for src_dir in (workspace_root, workspace_root / "src"):
        local_package_path = src_dir / "sam3"
        if local_package_path.exists() and str(local_package_path) not in sys.path:
            sys.path.insert(0, str(local_package_path))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SingleCameraSAMRopeNode(Node):
    def __init__(self):
        super().__init__("front_single_camera_sam_rope_node")

        self.declare_parameter(
            "sam_checkpoint_path",
            "/home/jeff/trustworthroboticsgroup/CoRL2026/deformables_ws/src/perception/checkpoints/sam3.pt",
        )
        self.declare_parameter("sam_prompt", "rope")
        self.declare_parameter("sam_confidence_threshold", 0.35)
        self.declare_parameter("run_sam_every_frame", True)
        self.declare_parameter("resize_width", 384)

        self.declare_parameter("num_keypoints", 64)
        self.declare_parameter("min_skeleton_pixels", 20)
        self.declare_parameter("morph_kernel_size", 5)

        self.sam_checkpoint_path = str(self.get_parameter("sam_checkpoint_path").value)
        self.sam_prompt = str(self.get_parameter("sam_prompt").value)
        self.sam_confidence_threshold = float(
            self.get_parameter("sam_confidence_threshold").value
        )
        self.declare_parameter("target_hz", 0.2)

        self.target_hz = float(self.get_parameter("target_hz").value)
        self.last_process_time = self.get_clock().now()

        self.run_sam_every_frame = bool(
            self.get_parameter("run_sam_every_frame").value
        )
        self.resize_width = int(self.get_parameter("resize_width").value)

        self.num_keypoints = int(self.get_parameter("num_keypoints").value)
        self.min_skeleton_pixels = int(self.get_parameter("min_skeleton_pixels").value)
        self.morph_kernel_size = int(self.get_parameter("morph_kernel_size").value)

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
            "/front_camera/color/image_raw",
            self.image_callback,
            1,
        )

        self.mask_pub = self.create_publisher(
            Image,
            "/front_camera/sam_rope/mask",
            1,
        )

        self.skeleton_pub = self.create_publisher(
            Image,
            "/front_camera/sam_rope/skeleton",
            1,
        )

        self.keypoints_pub = self.create_publisher(
            PoseArray,
            "/front_camera/sam_rope/keypoints",
            1,
        )

        self.annotated_pub = self.create_publisher(
            CompressedImage,
            "/front_camera/sam_rope/annotated_image/compressed",
            1,
        )
        self.stop_front_sub = self.create_subscription(
            Bool,
            '/stop_front',
            self.stop_callback,
            1
        )
        self.stop = False

    def stop_callback(self, msg):
        self.stop = msg.data

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

    def clean_mask(self, mask):
        mask_u8 = (mask > 0).astype(np.uint8)

        if self.morph_kernel_size <= 1:
            return mask_u8

        k = self.morph_kernel_size
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))

        cleaned = cv.morphologyEx(mask_u8, cv.MORPH_OPEN, kernel)
        cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
            cleaned,
            connectivity=8,
        )

        if num_labels <= 1:
            return cleaned

        largest_label = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
        cleaned = (labels == largest_label).astype(np.uint8)

        return cleaned

    def skeletonize_mask(self, claned_mask):
        # cleaned = self.clean_mask(mask)
        skel = skeletonize(claned_mask.astype(bool))
        return skel.astype(np.uint8)

    def get_neighbors(self, pixel, pixel_set):
        y, x = pixel
        neighbors = []

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue

                q = (y + dy, x + dx)
                if q in pixel_set:
                    neighbors.append(q)

        return neighbors

    def bfs_farthest(self, start, pixel_set):
        queue = deque([start])
        parent = {start: None}
        dist = {start: 0}

        farthest = start

        while queue:
            p = queue.popleft()

            if dist[p] > dist[farthest]:
                farthest = p

            for q in self.get_neighbors(p, pixel_set):
                if q not in parent:
                    parent[q] = p
                    dist[q] = dist[p] + 1
                    queue.append(q)

        return farthest, parent, dist

    def order_skeleton_pixels(self, skeleton):
        ys, xs = np.nonzero(skeleton > 0)

        if len(xs) < self.min_skeleton_pixels:
            return None

        pixels = list(zip(ys.tolist(), xs.tolist()))
        pixel_set = set(pixels)

        start = pixels[0]
        end_a, _, _ = self.bfs_farthest(start, pixel_set)
        end_b, parent, _ = self.bfs_farthest(end_a, pixel_set)

        path = []
        p = end_b

        while p is not None:
            path.append(p)
            p = parent[p]

        path.reverse()

        if len(path) < self.min_skeleton_pixels:
            return None

        points_xy = np.array([[x, y] for y, x in path], dtype=np.float32)
        return points_xy

    def sample_keypoints(self, ordered_points_xy):
        if ordered_points_xy is None or len(ordered_points_xy) == 0:
            return None

        if len(ordered_points_xy) == 1:
            return np.repeat(ordered_points_xy, self.num_keypoints, axis=0)

        diffs = np.diff(ordered_points_xy, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        total_length = cumulative[-1]

        if total_length < 1e-6:
            return None

        target = np.linspace(0.0, total_length, self.num_keypoints)
        sampled = np.zeros((self.num_keypoints, 2), dtype=np.float32)

        sampled[:, 0] = np.interp(target, cumulative, ordered_points_xy[:, 0])
        sampled[:, 1] = np.interp(target, cumulative, ordered_points_xy[:, 1])

        return sampled

    def publish_keypoints(self, keypoints_xy, header):
        msg = PoseArray()

        msg.header = header

        for i, p in enumerate(keypoints_xy):
            pose = Pose()

            pose.position.x = float(p[0])
            pose.position.y = float(p[1])

            # store keypoint index
            pose.position.z = float(i)

            msg.poses.append(pose)

        self.keypoints_pub.publish(msg)

    def draw_mask_overlay(self, rgb, mask, skeleton=None, keypoints_xy=None):
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

        if skeleton is not None:
            ys, xs = np.nonzero(skeleton > 0)
            vis[ys, xs] = np.array([255, 255, 0], dtype=np.uint8)

        if keypoints_xy is not None:
            for i, p in enumerate(keypoints_xy):
                x = int(round(p[0]))
                y = int(round(p[1]))

                cv.circle(vis, (x, y), 4, (255, 0, 255), -1)
                cv.putText(
                    vis,
                    str(i),
                    (x + 4, y - 4),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                )

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
        if self.stop:
            return

        try:
            now = self.get_clock().now()

            dt = (now - self.last_process_time).nanoseconds * 1e-9

            if dt < (1.0 / self.target_hz):
                return

            self.last_process_time = now
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

            mask_full = self.clean_mask(mask_full)
            skeleton = self.skeletonize_mask(mask_full)
            ordered_points = self.order_skeleton_pixels(skeleton)
            keypoints_xy = self.sample_keypoints(ordered_points)

            mask_msg = self.bridge.cv2_to_imgmsg(
                (mask_full * 255).astype(np.uint8),
                encoding="mono8",
            )
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)

            skeleton_msg = self.bridge.cv2_to_imgmsg(
                (skeleton * 255).astype(np.uint8),
                encoding="mono8",
            )
            skeleton_msg.header = msg.header
            self.skeleton_pub.publish(skeleton_msg)

            if keypoints_xy is not None:
                self.publish_keypoints(keypoints_xy, msg.header)
            else:
                self.get_logger().warn("Could not extract rope keypoints")

            annotated_rgb = self.draw_mask_overlay(
                frame_rgb,
                mask_full,
                skeleton=skeleton,
                keypoints_xy=keypoints_xy,
            )
            self.publish_annotated(annotated_rgb, msg.header)

            n_kp = 0 if keypoints_xy is None else len(keypoints_xy)
            self.get_logger().info(
                f"Published rope mask/skeleton/keypoints. "
                f"pixels={int(np.sum(mask_full > 0))}, keypoints={n_kp}"
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