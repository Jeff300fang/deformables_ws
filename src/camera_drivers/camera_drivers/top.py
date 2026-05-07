#!/usr/bin/env python3

import sys
import time
import uuid
import shutil
import tempfile
import traceback
from pathlib import Path
from collections import deque
from dataclasses import dataclass

import cv2 as cv
import numpy as np
import torch
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
import sensor_msgs_py.point_cloud2 as pc2
from skimage.morphology import skeletonize


for workspace_root in [Path.cwd(), *Path(__file__).resolve().parents]:
    for src_dir in (workspace_root, workspace_root / "src"):
        sam_path = src_dir / "sam3"
        if sam_path.exists() and str(sam_path) not in sys.path:
            sys.path.insert(0, str(sam_path))

from sam3.model_builder import build_sam3_multiplex_video_predictor


@dataclass
class CameraState:
    name: str
    image_topic: str
    depth_topic: str
    camera_info_topic: str
    annotated_topic: str
    mask_topic: str
    skeleton_topic: str
    pointcloud_topic: str

    fx: float = None
    fy: float = None
    cx: float = None
    cy: float = None

    rgb_frames: list = None
    depth_frames: list = None
    headers: list = None
    prev_points: np.ndarray = None

    def __post_init__(self):
        self.rgb_frames = []
        self.depth_frames = []
        self.headers = []


class SAM31RealSenseRopeDepthNode(Node):
    def __init__(self):
        super().__init__("sam31_realsense_rope_depth_node")

        self.declare_parameter("image_topic", "/realsense/camera/color/image_raw")
        self.declare_parameter(
            "depth_topic",
            "/realsense/camera/aligned_depth_to_color/image_raw",
        )
        self.declare_parameter(
            "camera_info_topic",
            "/realsense/camera/color/camera_info",
        )

        self.declare_parameter(
            "pointcloud_topic",
            "/tapnext/realsense/points_3d",
        )
        self.declare_parameter(
            "annotated_topic",
            "/sam31/realsense/annotated/compressed",
        )
        self.declare_parameter(
            "mask_topic",
            "/sam31/realsense/mask",
        )
        self.declare_parameter(
            "skeleton_topic",
            "/sam31/realsense/skeleton",
        )

        self.declare_parameter(
            "checkpoint_path",
            "/home/jeffreyfang/deformables/src/sam3/checkpoints/sam3.1_multiplex.pt",
        )

        self.declare_parameter("prompt", "rope")
        self.declare_parameter("confidence_threshold", 0.35)
        self.declare_parameter("process_every_n_frames", 1)
        self.declare_parameter("clip_length", 2)
        self.declare_parameter("offload_video_to_cpu", False)

        self.declare_parameter("max_skeleton_points", 80)
        self.declare_parameter("depth_search_radius", 4)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 5.0)
        self.declare_parameter("morph_kernel_size", 5)

        self.declare_parameter("temporal_alpha", 0.70)
        self.declare_parameter("max_point_jump_m", 0.08)
        self.declare_parameter("max_global_jump_m", 0.15)

        self.cam = CameraState(
            name="realsense",
            image_topic=str(self.get_parameter("image_topic").value),
            depth_topic=str(self.get_parameter("depth_topic").value),
            camera_info_topic=str(self.get_parameter("camera_info_topic").value),
            annotated_topic=str(self.get_parameter("annotated_topic").value),
            mask_topic=str(self.get_parameter("mask_topic").value),
            skeleton_topic=str(self.get_parameter("skeleton_topic").value),
            pointcloud_topic=str(self.get_parameter("pointcloud_topic").value),
        )

        self.checkpoint_path = str(
            self.get_parameter("checkpoint_path").value
        )

        self.prompt = str(self.get_parameter("prompt").value)

        self.confidence_threshold = float(
            self.get_parameter("confidence_threshold").value
        )

        self.process_every_n_frames = int(
            self.get_parameter("process_every_n_frames").value
        )

        self.clip_length = int(
            self.get_parameter("clip_length").value
        )

        self.offload_video_to_cpu = bool(
            self.get_parameter("offload_video_to_cpu").value
        )

        self.max_skeleton_points = int(
            self.get_parameter("max_skeleton_points").value
        )

        self.depth_search_radius = int(
            self.get_parameter("depth_search_radius").value
        )

        self.min_depth_m = float(
            self.get_parameter("min_depth_m").value
        )

        self.max_depth_m = float(
            self.get_parameter("max_depth_m").value
        )

        self.morph_kernel_size = int(
            self.get_parameter("morph_kernel_size").value
        )

        self.temporal_alpha = float(
            self.get_parameter("temporal_alpha").value
        )

        self.max_point_jump_m = float(
            self.get_parameter("max_point_jump_m").value
        )

        self.max_global_jump_m = float(
            self.get_parameter("max_global_jump_m").value
        )

        self.bridge = CvBridge()

        self.frame_count = 0
        self.processing = False

        if not torch.cuda.is_available():
            raise RuntimeError(
                "SAM 3.1 multiplex video predictor requires CUDA."
            )

        self.get_logger().info(
            "Loading SAM 3.1 multiplex video predictor..."
        )

        self.predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=self.checkpoint_path,
            default_output_prob_thresh=self.confidence_threshold,
            async_loading_frames=True,
            use_fa3=False,
        )

        self.rgb_sub = Subscriber(
            self,
            Image,
            self.cam.image_topic,
        )

        self.depth_sub = Subscriber(
            self,
            Image,
            self.cam.depth_topic,
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.08,
        )

        self.sync.registerCallback(self.synced_callback)

        self.info_sub = self.create_subscription(
            CameraInfo,
            self.cam.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.annotated_pub = self.create_publisher(
            CompressedImage,
            self.cam.annotated_topic,
            10,
        )

        self.mask_pub = self.create_publisher(
            Image,
            self.cam.mask_topic,
            10,
        )

        self.skeleton_pub = self.create_publisher(
            Image,
            self.cam.skeleton_topic,
            10,
        )

        self.cloud_pub = self.create_publisher(
            PointCloud2,
            self.cam.pointcloud_topic,
            10,
        )

        self.get_logger().info(
            "Single RealSense SAM3D rope node started."
        )

        self.get_logger().info(f"RGB: {self.cam.image_topic}")
        self.get_logger().info(f"Depth: {self.cam.depth_topic}")
        self.get_logger().info(
            f"Camera info: {self.cam.camera_info_topic}"
        )

        self.get_logger().info(
            f"Point cloud: {self.cam.pointcloud_topic}"
        )

    def camera_info_callback(self, msg: CameraInfo):
        self.cam.fx = float(msg.k[0])
        self.cam.fy = float(msg.k[4])
        self.cam.cx = float(msg.k[2])
        self.cam.cy = float(msg.k[5])

    def synced_callback(self, rgb_msg, depth_msg):
        if self.processing:
            return

        self.frame_count += 1

        if self.frame_count % self.process_every_n_frames != 0:
            return

        try:
            self.append_frame(rgb_msg, depth_msg)

            if len(self.cam.rgb_frames) >= self.clip_length:
                self.processing = True
                self.process_clip()

        except Exception as e:
            self.processing = False
            self.get_logger().error(
                f"Synced callback failed: {e}"
            )
            self.get_logger().error(traceback.format_exc())

    def append_frame(self, rgb_msg: Image, depth_msg: Image):
        frame_bgr = self.bridge.imgmsg_to_cv2(
            rgb_msg,
            desired_encoding="bgr8",
        )

        depth_img = self.bridge.imgmsg_to_cv2(
            depth_msg,
            desired_encoding="passthrough",
        )

        depth_img = np.asarray(depth_img)

        self.cam.rgb_frames.append(frame_bgr.copy())
        self.cam.depth_frames.append(depth_img.copy())
        self.cam.headers.append(rgb_msg.header)

    def process_clip(self):
        try:
            frames, depths, headers = self.pop_buffers()

            results = self.run_sam_clip(
                frames,
                depths,
                headers,
            )

            for result in results:
                self.publish_pointcloud(
                    result["points_3d"],
                    result["header"],
                    self.cloud_pub,
                )

                self.publish_annotated(
                    result["annotated_bgr"],
                    result["header"],
                    self.annotated_pub,
                )

                self.publish_mask(
                    result["clean_mask"],
                    result["header"],
                    self.mask_pub,
                )

                self.publish_mask(
                    result["skeleton"].astype(np.uint8) * 255,
                    result["header"],
                    self.skeleton_pub,
                )

            self.get_logger().info(
                f"Published RealSense SAM3D results for {len(results)} frames"
            )

        except Exception as e:
            self.get_logger().error(
                f"Clip processing failed: {e}"
            )
            self.get_logger().error(traceback.format_exc())

        finally:
            self.processing = False

    def pop_buffers(self):
        frames = self.cam.rgb_frames
        depths = self.cam.depth_frames
        headers = self.cam.headers

        self.cam.rgb_frames = []
        self.cam.depth_frames = []
        self.cam.headers = []

        return frames, depths, headers

    def run_sam_clip(self, rgb_frames, depth_frames, headers):
        clip_dir = Path(
            tempfile.mkdtemp(prefix="sam31_realsense_clip_")
        )

        session_id = None

        try:
            for idx, frame_bgr in enumerate(rgb_frames):
                cv.imwrite(
                    str(clip_dir / f"{idx:05d}.jpg"),
                    frame_bgr,
                )

            session_id = self.start_video_session(str(clip_dir))

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
                outputs_by_frame[
                    response["frame_index"]
                ] = response["outputs"]

            results = []

            for idx, frame_bgr in enumerate(rgb_frames):
                header = headers[idx]
                depth_img = depth_frames[idx]

                outputs = outputs_by_frame.get(idx, {})

                masks = self.extract_masks(
                    outputs,
                    frame_bgr.shape[:2],
                )

                binary_mask = self.combine_masks(
                    masks,
                    frame_bgr.shape[:2],
                )

                clean_mask = self.clean_binary_mask(binary_mask)

                skeleton = self.skeletonize_mask(clean_mask)

                ordered_skeleton_xy = self.order_skeleton_points_graph(
                    skeleton
                )

                sampled_xy = self.sample_ordered_points(
                    ordered_skeleton_xy,
                    self.max_skeleton_points,
                )

                points_3d, valid_xy, depths_m = (
                    self.skeleton_depth_to_points(
                        sampled_xy,
                        depth_img,
                    )
                )

                points_3d = self.temporal_filter_points(
                    points_3d
                )

                annotated_bgr = self.draw_result(
                    frame_bgr,
                    clean_mask,
                    skeleton,
                    valid_xy,
                    depths_m,
                )

                results.append(
                    {
                        "header": header,
                        "points_3d": points_3d,
                        "valid_xy": valid_xy,
                        "depths_m": depths_m,
                        "clean_mask": clean_mask,
                        "skeleton": skeleton,
                        "annotated_bgr": annotated_bgr,
                    }
                )

            return results

        finally:
            if session_id is not None:
                self.predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=session_id,
                    )
                )

            shutil.rmtree(clip_dir, ignore_errors=True)

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
            return np.zeros(
                (0, frame_shape[0], frame_shape[1]),
                dtype=bool,
            )

        if torch.is_tensor(masks):
            masks = masks.detach().cpu().numpy()
        else:
            masks = np.asarray(masks)

        masks = np.squeeze(masks)

        if masks.ndim == 2:
            masks = masks[None]

        if masks.size == 0:
            return np.zeros(
                (0, frame_shape[0], frame_shape[1]),
                dtype=bool,
            )

        masks = masks.astype(bool)

        h, w = frame_shape

        resized_masks = []

        for mask in masks:
            if mask.shape[:2] != (h, w):
                mask_u8 = mask.astype(np.uint8) * 255

                mask_u8 = cv.resize(
                    mask_u8,
                    (w, h),
                    interpolation=cv.INTER_NEAREST,
                )

                mask = mask_u8 > 0

            resized_masks.append(mask)

        return np.asarray(resized_masks, dtype=bool)

    def combine_masks(self, masks, frame_shape):
        if masks.shape[0] == 0:
            return np.zeros(frame_shape, dtype=np.uint8)

        return np.any(masks, axis=0).astype(np.uint8) * 255

    def clean_binary_mask(self, binary_mask):
        mask = (binary_mask > 0).astype(np.uint8) * 255

        k = max(3, self.morph_kernel_size)

        if k % 2 == 0:
            k += 1

        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE,
            (k, k),
        )

        mask = cv.morphologyEx(
            mask,
            cv.MORPH_CLOSE,
            kernel,
        )

        mask = cv.morphologyEx(
            mask,
            cv.MORPH_OPEN,
            kernel,
        )

        num_labels, labels, stats, _ = (
            cv.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        if num_labels <= 1:
            return mask

        largest_label = (
            1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
        )

        clean = np.zeros_like(mask)
        clean[labels == largest_label] = 255

        return clean

    def skeletonize_mask(self, binary_mask):
        mask_bool = binary_mask > 0

        if np.sum(mask_bool) < 20:
            return np.zeros_like(mask_bool, dtype=bool)

        return skeletonize(mask_bool).astype(bool)

    def order_skeleton_points_graph(self, skeleton):
        ys, xs = np.where(skeleton)

        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float32)

        nodes = set(
            (int(x), int(y))
            for x, y in zip(xs, ys)
        )

        if len(nodes) <= 2:
            return np.asarray(list(nodes), dtype=np.float32)

        def neighbors(p, valid_nodes):
            x, y = p

            out = []

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue

                    q = (x + dx, y + dy)

                    if q in valid_nodes:
                        out.append(q)

            return out

        visited = set()
        components = []

        for node in nodes:
            if node in visited:
                continue

            q = deque([node])
            visited.add(node)

            comp = []

            while q:
                p = q.popleft()
                comp.append(p)

                for nb in neighbors(p, nodes):
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)

            components.append(comp)

        nodes = set(max(components, key=len))

        def bfs_farthest(start):
            q = deque([start])

            parent = {start: None}
            dist = {start: 0}

            farthest = start

            while q:
                p = q.popleft()

                if dist[p] > dist[farthest]:
                    farthest = p

                for nb in neighbors(p, nodes):
                    if nb not in parent:
                        parent[nb] = p
                        dist[nb] = dist[p] + 1
                        q.append(nb)

            return farthest, parent

        start = next(iter(nodes))

        a, _ = bfs_farthest(start)
        b, parent = bfs_farthest(a)

        path = []

        cur = b

        while cur is not None:
            path.append(cur)
            cur = parent[cur]

        return np.asarray(path[::-1], dtype=np.float32)

    def sample_ordered_points(self, ordered_xy, max_points):
        if (
            ordered_xy.shape[0] == 0
            or ordered_xy.shape[0] <= max_points
        ):
            return ordered_xy

        idx = np.linspace(
            0,
            ordered_xy.shape[0] - 1,
            max_points,
        ).astype(np.int32)

        return ordered_xy[idx]

    def depth_to_meters(self, depth_img):
        if depth_img.dtype == np.uint16:
            return depth_img.astype(np.float32) * 0.001

        return depth_img.astype(np.float32)

    def query_depth_near_pixel(self, depth_m, x, y):
        h, w = depth_m.shape[:2]

        r = self.depth_search_radius

        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)

        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)

        patch = depth_m[y0:y1, x0:x1]

        if patch.size == 0:
            return np.nan

        valid = (
            np.isfinite(patch)
            & (patch >= self.min_depth_m)
            & (patch <= self.max_depth_m)
        )

        if not np.any(valid):
            return np.nan

        vals = patch[valid].astype(np.float32)

        return float(np.percentile(vals, 15.0))

    def filter_depth_along_centerline(
        self,
        valid_xy,
        depths,
        max_jump_m=0.08,
        mad_scale=2.5,
    ):
        if depths.shape[0] < 5:
            return valid_xy, depths

        finite = np.isfinite(depths) & (depths > 0.0)

        valid_xy = valid_xy[finite]
        depths = depths[finite]

        if depths.shape[0] < 5:
            return valid_xy, depths

        med = np.median(depths)

        mad = (
            np.median(np.abs(depths - med))
            + 1e-6
        )

        robust_keep = (
            np.abs(depths - med)
            <= mad_scale * 1.4826 * mad
        )

        jump_keep = np.ones_like(depths, dtype=bool)

        for i in range(1, depths.shape[0]):
            if abs(depths[i] - depths[i - 1]) > max_jump_m:
                if depths[i] > depths[i - 1]:
                    jump_keep[i] = False
                else:
                    jump_keep[i - 1] = False

        keep = robust_keep & jump_keep

        return valid_xy[keep], depths[keep]

    def skeleton_depth_to_points(self, sampled_xy, depth_img):
        if (
            self.cam.fx is None
            or self.cam.fy is None
            or self.cam.cx is None
            or self.cam.cy is None
        ):
            self.get_logger().warn(
                "Camera intrinsics not received yet."
            )

            return self.empty_points_return()

        if sampled_xy.shape[0] == 0:
            return self.empty_points_return()

        depth_m = self.depth_to_meters(depth_img)

        h, w = depth_m.shape[:2]

        valid_xy = []
        depths = []

        for xy in sampled_xy:
            x = int(round(float(xy[0])))
            y = int(round(float(xy[1])))

            if x < 0 or x >= w or y < 0 or y >= h:
                continue

            z = self.query_depth_near_pixel(
                depth_m,
                x,
                y,
            )

            if np.isfinite(z) and z > 0.0:
                valid_xy.append([x, y])
                depths.append(z)

        if len(valid_xy) == 0:
            return self.empty_points_return()

        valid_xy = np.asarray(valid_xy, dtype=np.float32)
        depths = np.asarray(depths, dtype=np.float32)

        valid_xy, depths = self.filter_depth_along_centerline(
            valid_xy,
            depths,
        )

        if depths.shape[0] == 0:
            return self.empty_points_return()

        xs = valid_xy[:, 0]
        ys = valid_xy[:, 1]
        zs = depths

        x_c = (xs - self.cam.cx) * zs / self.cam.fx
        y_c = (ys - self.cam.cy) * zs / self.cam.fy
        z_c = zs

        points = np.stack(
            [x_c, y_c, z_c],
            axis=1,
        ).astype(np.float32)

        return points, valid_xy, depths

    def empty_points_return(self):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    def temporal_filter_points(self, points_now):
        points_now = np.asarray(
            points_now,
            dtype=np.float32,
        )

        if points_now.shape[0] == 0:
            if self.cam.prev_points is None:
                return points_now

            return self.cam.prev_points.copy()

        if (
            self.cam.prev_points is None
            or self.cam.prev_points.shape != points_now.shape
        ):
            self.cam.prev_points = points_now.copy()
            return points_now

        delta = np.linalg.norm(
            points_now - self.cam.prev_points,
            axis=1,
        )

        global_jump = float(np.median(delta))

        if global_jump > self.max_global_jump_m:
            return self.cam.prev_points.copy()

        filtered = (
            self.temporal_alpha * self.cam.prev_points
            + (1.0 - self.temporal_alpha) * points_now
        )

        jump_mask = delta > self.max_point_jump_m

        filtered[jump_mask] = self.cam.prev_points[jump_mask]

        self.cam.prev_points = filtered.copy()

        return filtered

    def draw_result(
        self,
        frame_bgr,
        mask,
        skeleton,
        valid_xy,
        depths_m,
    ):
        vis = frame_bgr.copy()

        overlay = vis.copy()

        overlay[mask > 0] = (
            0.65 * overlay[mask > 0]
            + 0.35 * np.array([0, 0, 255])
        ).astype(np.uint8)

        vis = overlay

        ys, xs = np.where(skeleton)

        for x, y in zip(xs, ys):
            cv.circle(
                vis,
                (int(x), int(y)),
                1,
                (0, 255, 255),
                -1,
            )

        for i, (xy, d) in enumerate(
            zip(valid_xy, depths_m)
        ):
            x = int(round(float(xy[0])))
            y = int(round(float(xy[1])))

            cv.circle(
                vis,
                (x, y),
                5,
                (0, 255, 0),
                -1,
            )

            cv.putText(
                vis,
                f"realsense:{i} {d:.3f}m",
                (x + 6, y - 6),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        return vis

    def publish_annotated(self, bgr_img, header, pub):
        ok, encoded = cv.imencode(
            ".jpg",
            bgr_img,
            [int(cv.IMWRITE_JPEG_QUALITY), 90],
        )

        if not ok:
            return

        msg = CompressedImage()
        msg.header = header
        msg.format = "jpeg"
        msg.data = encoded.tobytes()

        pub.publish(msg)

    def publish_mask(self, mask_u8, header, pub):
        msg = self.bridge.cv2_to_imgmsg(
            mask_u8.astype(np.uint8),
            encoding="mono8",
        )

        msg.header = header

        pub.publish(msg)

    def publish_pointcloud(self, points_3d, header, pub):
        msg = pc2.create_cloud_xyz32(
            header,
            points_3d.astype(np.float32).tolist(),
        )

        pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = SAM31RealSenseRopeDepthNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()