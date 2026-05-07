#!/usr/bin/env python3

import sys
import time
import uuid
import json
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


def load_cam2_to_cam1(calib_path):
    with open(calib_path, "r") as f:
        calib = json.load(f)

    R = np.array(calib["camera_poses"]["cam2_to_cam1"]["R"], dtype=np.float64)
    T = np.array(calib["camera_poses"]["cam2_to_cam1"]["T"], dtype=np.float64)

    T_cam2_to_cam1 = np.eye(4, dtype=np.float64)
    T_cam2_to_cam1[:3, :3] = R
    T_cam2_to_cam1[:3, 3] = T
    return T_cam2_to_cam1


class SAM31TwoCameraRopeDepthNode(Node):
    def __init__(self):
        super().__init__("sam31_two_camera_rope_depth_node")

        self.declare_parameter("cam1_image_topic", "/front_camera/color/image_raw")
        self.declare_parameter("cam1_depth_topic", "/front_camera/depth/image_raw")
        self.declare_parameter("cam1_camera_info_topic", "/front_camera/color/camera_info")

        self.declare_parameter("cam2_image_topic", "/back_camera/color/image_raw")
        self.declare_parameter("cam2_depth_topic", "/back_camera/depth/image_raw")
        self.declare_parameter("cam2_camera_info_topic", "/back_camera/color/camera_info")

        self.declare_parameter("global_frame", "cam1_frame")
        self.declare_parameter("fused_pointcloud_topic", "/tapnext/fused_points_3d")

        self.declare_parameter("cam1_pointcloud_topic", "/tapnext/cam1/points_3d")
        self.declare_parameter("cam2_pointcloud_topic", "/tapnext/cam2/points_3d")

        self.declare_parameter("cam1_annotated_topic", "/sam31/cam1/annotated/compressed")
        self.declare_parameter("cam2_annotated_topic", "/sam31/cam2/annotated/compressed")
        self.declare_parameter("cam1_mask_topic", "/sam31/cam1/mask")
        self.declare_parameter("cam2_mask_topic", "/sam31/cam2/mask")
        self.declare_parameter("cam1_skeleton_topic", "/sam31/cam1/skeleton")
        self.declare_parameter("cam2_skeleton_topic", "/sam31/cam2/skeleton")

        self.declare_parameter(
            "checkpoint_path",
            "/home/jeffreyfang/deformables/src/sam3/checkpoints/sam3.1_multiplex.pt",
        )
        self.declare_parameter("prompt", "rope")
        self.declare_parameter("confidence_threshold", 0.35)
        self.declare_parameter("process_every_n_frames", 1)
        self.declare_parameter("clip_length", 2)
        self.declare_parameter("loop", True)
        self.declare_parameter("offload_video_to_cpu", False)

        self.declare_parameter("max_skeleton_points", 80)
        self.declare_parameter("depth_search_radius", 4)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 5.0)
        self.declare_parameter("morph_kernel_size", 5)

        self.declare_parameter("temporal_alpha", 0.70)
        self.declare_parameter("max_point_jump_m", 0.08)
        self.declare_parameter("max_global_jump_m", 0.15)

        self.declare_parameter("calib_path", "/home/jeffreyfang/calib/dataset/calibration.json")
        self.declare_parameter("use_calibration_base", True)
        self.declare_parameter("cam2_tx", -0.32)
        self.declare_parameter("cam2_ty", 0.1)
        self.declare_parameter("cam2_tz", 1.44)
        self.declare_parameter("cam2_roll", 3.14)
        self.declare_parameter("cam2_pitch", -0.55)
        self.declare_parameter("cam2_yaw", -0.15)

        self.global_frame = str(self.get_parameter("global_frame").value)
        self.fused_pointcloud_topic = str(self.get_parameter("fused_pointcloud_topic").value)

        self.cam1 = CameraState(
            name="cam1",
            image_topic=str(self.get_parameter("cam1_image_topic").value),
            depth_topic=str(self.get_parameter("cam1_depth_topic").value),
            camera_info_topic=str(self.get_parameter("cam1_camera_info_topic").value),
            annotated_topic=str(self.get_parameter("cam1_annotated_topic").value),
            mask_topic=str(self.get_parameter("cam1_mask_topic").value),
            skeleton_topic=str(self.get_parameter("cam1_skeleton_topic").value),
            pointcloud_topic=str(self.get_parameter("cam1_pointcloud_topic").value),
        )

        self.cam2 = CameraState(
            name="cam2",
            image_topic=str(self.get_parameter("cam2_image_topic").value),
            depth_topic=str(self.get_parameter("cam2_depth_topic").value),
            camera_info_topic=str(self.get_parameter("cam2_camera_info_topic").value),
            annotated_topic=str(self.get_parameter("cam2_annotated_topic").value),
            mask_topic=str(self.get_parameter("cam2_mask_topic").value),
            skeleton_topic=str(self.get_parameter("cam2_skeleton_topic").value),
            pointcloud_topic=str(self.get_parameter("cam2_pointcloud_topic").value),
        )

        self.checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        self.prompt = str(self.get_parameter("prompt").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.process_every_n_frames = int(self.get_parameter("process_every_n_frames").value)
        self.clip_length = int(self.get_parameter("clip_length").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.offload_video_to_cpu = bool(self.get_parameter("offload_video_to_cpu").value)

        self.max_skeleton_points = int(self.get_parameter("max_skeleton_points").value)
        self.depth_search_radius = int(self.get_parameter("depth_search_radius").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.morph_kernel_size = int(self.get_parameter("morph_kernel_size").value)

        self.temporal_alpha = float(self.get_parameter("temporal_alpha").value)
        self.max_point_jump_m = float(self.get_parameter("max_point_jump_m").value)
        self.max_global_jump_m = float(self.get_parameter("max_global_jump_m").value)

        self.bridge = CvBridge()
        self.frame_count = 0
        self.processing = False

        self.T_cam1_to_cam1 = np.eye(4, dtype=np.float64)

        calib_path = str(self.get_parameter("calib_path").value)
        if Path(calib_path).exists():
            self.T_cam2_to_cam1_calib = load_cam2_to_cam1(calib_path)
            self.get_logger().info(f"Loaded cam2->cam1 calibration: {calib_path}")
        else:
            self.T_cam2_to_cam1_calib = np.eye(4, dtype=np.float64)
            self.get_logger().warn(f"Calibration file not found: {calib_path}. Using identity.")

        if not torch.cuda.is_available():
            raise RuntimeError("SAM 3.1 multiplex video predictor requires CUDA.")

        self.get_logger().info("Loading SAM 3.1 multiplex video predictor...")
        self.predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=self.checkpoint_path,
            default_output_prob_thresh=self.confidence_threshold,
            async_loading_frames=True,
            use_fa3=False,
        )

        self.cam1_rgb_sub = Subscriber(self, Image, self.cam1.image_topic)
        self.cam1_depth_sub = Subscriber(self, Image, self.cam1.depth_topic)
        self.cam2_rgb_sub = Subscriber(self, Image, self.cam2.image_topic)
        self.cam2_depth_sub = Subscriber(self, Image, self.cam2.depth_topic)

        self.sync = ApproximateTimeSynchronizer(
            [
                self.cam1_rgb_sub,
                self.cam1_depth_sub,
                self.cam2_rgb_sub,
                self.cam2_depth_sub,
            ],
            queue_size=10,
            slop=0.08,
        )
        self.sync.registerCallback(self.synced_callback)

        self.cam1_info_sub = self.create_subscription(
            CameraInfo,
            self.cam1.camera_info_topic,
            lambda msg: self.camera_info_callback(msg, self.cam1),
            10,
        )
        self.cam2_info_sub = self.create_subscription(
            CameraInfo,
            self.cam2.camera_info_topic,
            lambda msg: self.camera_info_callback(msg, self.cam2),
            10,
        )

        self.cam1_annotated_pub = self.create_publisher(CompressedImage, self.cam1.annotated_topic, 10)
        self.cam2_annotated_pub = self.create_publisher(CompressedImage, self.cam2.annotated_topic, 10)

        self.cam1_mask_pub = self.create_publisher(Image, self.cam1.mask_topic, 10)
        self.cam2_mask_pub = self.create_publisher(Image, self.cam2.mask_topic, 10)

        self.cam1_skeleton_pub = self.create_publisher(Image, self.cam1.skeleton_topic, 10)
        self.cam2_skeleton_pub = self.create_publisher(Image, self.cam2.skeleton_topic, 10)

        self.cam1_cloud_pub = self.create_publisher(PointCloud2, self.cam1.pointcloud_topic, 10)
        self.cam2_cloud_pub = self.create_publisher(PointCloud2, self.cam2.pointcloud_topic, 10)
        self.fused_cloud_pub = self.create_publisher(PointCloud2, self.fused_pointcloud_topic, 10)

        self.get_logger().info("Two-camera SAM3D rope node started.")
        self.get_logger().info(f"cam1 RGB: {self.cam1.image_topic}")
        self.get_logger().info(f"cam2 RGB: {self.cam2.image_topic}")
        self.get_logger().info(f"Global frame: {self.global_frame}")
        self.get_logger().info(f"Fused cloud: {self.fused_pointcloud_topic}")

    def camera_info_callback(self, msg: CameraInfo, cam: CameraState):
        cam.fx = float(msg.k[0])
        cam.fy = float(msg.k[4])
        cam.cx = float(msg.k[2])
        cam.cy = float(msg.k[5])

    def synced_callback(self, cam1_rgb_msg, cam1_depth_msg, cam2_rgb_msg, cam2_depth_msg):
        if self.processing:
            return

        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return

        try:
            self.append_frame(self.cam1, cam1_rgb_msg, cam1_depth_msg)
            self.append_frame(self.cam2, cam2_rgb_msg, cam2_depth_msg)

            if len(self.cam1.rgb_frames) >= self.clip_length and len(self.cam2.rgb_frames) >= self.clip_length:
                self.processing = True
                self.process_both_clips()

        except Exception as e:
            self.processing = False
            self.get_logger().error(f"Synced callback failed: {e}")
            self.get_logger().error(traceback.format_exc())

    def append_frame(self, cam: CameraState, rgb_msg: Image, depth_msg: Image):
        frame_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth_img = np.asarray(depth_img)

        cam.rgb_frames.append(frame_bgr.copy())
        cam.depth_frames.append(depth_img.copy())
        cam.headers.append(rgb_msg.header)

    def process_both_clips(self):
        try:
            cam1_frames, cam1_depths, cam1_headers = self.pop_cam_buffers(self.cam1)
            cam2_frames, cam2_depths, cam2_headers = self.pop_cam_buffers(self.cam2)

            cam1_results = self.run_sam_clip_for_camera(
                self.cam1,
                cam1_frames,
                cam1_depths,
                cam1_headers,
            )
            cam2_results = self.run_sam_clip_for_camera(
                self.cam2,
                cam2_frames,
                cam2_depths,
                cam2_headers,
            )

            T_cam2_live = self.get_cam2_transform_live()

            n = min(len(cam1_results), len(cam2_results))

            for i in range(n):
                r1 = cam1_results[i]
                r2 = cam2_results[i]

                header = r1["header"]
                global_header = self.copy_header_with_frame(header, self.global_frame)

                cam1_points_global = self.transform_points(r1["points_3d"], self.T_cam1_to_cam1)
                cam2_points_global = self.transform_points(r2["points_3d"], T_cam2_live)

                self.publish_pointcloud(cam1_points_global, global_header, self.cam1_cloud_pub)
                self.publish_pointcloud(cam2_points_global, global_header, self.cam2_cloud_pub)

                if cam1_points_global.shape[0] == 0 and cam2_points_global.shape[0] == 0:
                    fused = np.empty((0, 3), dtype=np.float32)
                elif cam1_points_global.shape[0] == 0:
                    fused = cam2_points_global
                elif cam2_points_global.shape[0] == 0:
                    fused = cam1_points_global
                else:
                    fused = np.concatenate([cam1_points_global, cam2_points_global], axis=0)

                self.publish_pointcloud(fused, global_header, self.fused_cloud_pub)

                self.publish_annotated(r1["annotated_bgr"], r1["header"], self.cam1_annotated_pub)
                self.publish_annotated(r2["annotated_bgr"], r2["header"], self.cam2_annotated_pub)

                self.publish_mask(r1["clean_mask"], r1["header"], self.cam1_mask_pub)
                self.publish_mask(r2["clean_mask"], r2["header"], self.cam2_mask_pub)

                self.publish_mask(r1["skeleton"].astype(np.uint8) * 255, r1["header"], self.cam1_skeleton_pub)
                self.publish_mask(r2["skeleton"].astype(np.uint8) * 255, r2["header"], self.cam2_skeleton_pub)

            self.get_logger().info(
                f"Published two-camera SAM3D results for {n} synchronized frames"
            )

        except Exception as e:
            self.get_logger().error(f"Two-camera clip processing failed: {e}")
            self.get_logger().error(traceback.format_exc())

        finally:
            self.processing = False

    def pop_cam_buffers(self, cam: CameraState):
        frames = cam.rgb_frames
        depths = cam.depth_frames
        headers = cam.headers

        cam.rgb_frames = []
        cam.depth_frames = []
        cam.headers = []

        return frames, depths, headers

    def run_sam_clip_for_camera(self, cam: CameraState, rgb_frames, depth_frames, headers):
        clip_dir = Path(tempfile.mkdtemp(prefix=f"sam31_{cam.name}_clip_"))
        session_id = None

        try:
            for idx, frame_bgr in enumerate(rgb_frames):
                cv.imwrite(str(clip_dir / f"{idx:05d}.jpg"), frame_bgr)

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

            outputs_by_frame = {response["frame_index"]: response["outputs"]}

            for response in self.predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    output_prob_thresh=self.confidence_threshold,
                )
            ):
                outputs_by_frame[response["frame_index"]] = response["outputs"]

            results = []

            for idx, frame_bgr in enumerate(rgb_frames):
                header = headers[idx]
                depth_img = depth_frames[idx]

                outputs = outputs_by_frame.get(idx, {})
                masks = self.extract_masks(outputs, frame_bgr.shape[:2])
                binary_mask = self.combine_masks(masks, frame_bgr.shape[:2])
                clean_mask = self.clean_binary_mask(binary_mask)
                skeleton = self.skeletonize_mask(clean_mask)

                ordered_skeleton_xy = self.order_skeleton_points_graph(skeleton)
                sampled_xy = self.sample_ordered_points(
                    ordered_skeleton_xy,
                    self.max_skeleton_points,
                )

                points_3d, valid_xy, depths_m = self.skeleton_depth_to_points(
                    cam,
                    sampled_xy,
                    depth_img,
                )

                points_3d = self.temporal_filter_points_for_camera(
                    cam,
                    points_3d,
                    alpha=self.temporal_alpha,
                    max_point_jump_m=self.max_point_jump_m,
                    max_global_jump_m=self.max_global_jump_m,
                )

                annotated_bgr = self.draw_result(
                    frame_bgr,
                    clean_mask,
                    skeleton,
                    valid_xy,
                    depths_m,
                    cam.name,
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

    def get_manual_cam2_transform(self):
        tx = float(self.get_parameter("cam2_tx").value)
        ty = float(self.get_parameter("cam2_ty").value)
        tz = float(self.get_parameter("cam2_tz").value)

        roll = float(self.get_parameter("cam2_roll").value)
        pitch = float(self.get_parameter("cam2_pitch").value)
        yaw = float(self.get_parameter("cam2_yaw").value)

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        Rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cr, -sr],
                [0.0, sr, cr],
            ],
            dtype=np.float64,
        )
        Ry = np.array(
            [
                [cp, 0.0, sp],
                [0.0, 1.0, 0.0],
                [-sp, 0.0, cp],
            ],
            dtype=np.float64,
        )
        Rz = np.array(
            [
                [cy, -sy, 0.0],
                [sy, cy, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        R = Rz @ Ry @ Rx

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
        return T

    def get_cam2_transform_live(self):
        T_manual = self.get_manual_cam2_transform()
        use_calibration_base = bool(self.get_parameter("use_calibration_base").value)

        if use_calibration_base:
            return T_manual @ self.T_cam2_to_cam1_calib

        return T_manual

    def transform_points(self, points, T):
        if points.shape[0] == 0:
            return points

        ones = np.ones((points.shape[0], 1), dtype=np.float64)
        points_h = np.concatenate([points.astype(np.float64), ones], axis=1)
        out_h = (T @ points_h.T).T
        return out_h[:, :3].astype(np.float32)

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

        masks = masks.astype(bool)

        resized_masks = []
        h, w = frame_shape

        for mask in masks:
            if mask.shape[:2] != (h, w):
                mask_u8 = mask.astype(np.uint8) * 255
                mask_u8 = cv.resize(mask_u8, (w, h), interpolation=cv.INTER_NEAREST)
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

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        if num_labels <= 1:
            return mask

        largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
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

        nodes = set((int(x), int(y)) for x, y in zip(xs, ys))

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

        largest = max(components, key=len)
        nodes = set(largest)

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

        path = path[::-1]
        return np.asarray(path, dtype=np.float32)

    def sample_ordered_points(self, ordered_xy, max_points):
        if ordered_xy.shape[0] == 0:
            return ordered_xy

        if ordered_xy.shape[0] <= max_points:
            return ordered_xy

        idx = np.linspace(0, ordered_xy.shape[0] - 1, max_points).astype(np.int32)
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

    def filter_depth_along_centerline(self, valid_xy, depths, max_jump_m=0.08, mad_scale=2.5):
        if depths.shape[0] < 5:
            return valid_xy, depths

        finite = np.isfinite(depths) & (depths > 0.0)
        valid_xy = valid_xy[finite]
        depths = depths[finite]

        if depths.shape[0] < 5:
            return valid_xy, depths

        med = np.median(depths)
        mad = np.median(np.abs(depths - med)) + 1e-6

        robust_keep = np.abs(depths - med) <= mad_scale * 1.4826 * mad
        jump_keep = np.ones_like(depths, dtype=bool)

        for i in range(1, depths.shape[0]):
            if abs(depths[i] - depths[i - 1]) > max_jump_m:
                if depths[i] > depths[i - 1]:
                    jump_keep[i] = False
                else:
                    jump_keep[i - 1] = False

        keep = robust_keep & jump_keep
        return valid_xy[keep], depths[keep]

    def skeleton_depth_to_points(self, cam: CameraState, sampled_xy, depth_img):
        if cam.fx is None or cam.fy is None or cam.cx is None or cam.cy is None:
            self.get_logger().warn(f"{cam.name}: camera intrinsics not received yet.")
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

            z = self.query_depth_near_pixel(depth_m, x, y)

            if not np.isfinite(z) or z <= 0.0:
                continue

            valid_xy.append([x, y])
            depths.append(z)

        if len(valid_xy) == 0:
            return self.empty_points_return()

        valid_xy = np.asarray(valid_xy, dtype=np.float32)
        depths = np.asarray(depths, dtype=np.float32)

        valid_xy, depths = self.filter_depth_along_centerline(
            valid_xy,
            depths,
            max_jump_m=0.08,
            mad_scale=2.5,
        )

        if depths.shape[0] == 0:
            return self.empty_points_return()

        xs = valid_xy[:, 0]
        ys = valid_xy[:, 1]
        zs = depths

        x_c = (xs - cam.cx) * zs / cam.fx
        y_c = (ys - cam.cy) * zs / cam.fy
        z_c = zs

        points = np.stack([x_c, y_c, z_c], axis=1).astype(np.float32)
        return points, valid_xy, depths

    def empty_points_return(self):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    def temporal_filter_points_for_camera(
        self,
        cam: CameraState,
        points_now,
        alpha=0.70,
        max_point_jump_m=0.08,
        max_global_jump_m=0.15,
    ):
        points_now = np.asarray(points_now, dtype=np.float32)

        if points_now.shape[0] == 0:
            if cam.prev_points is None:
                return points_now
            return cam.prev_points.copy()

        if cam.prev_points is None:
            cam.prev_points = points_now.copy()
            return points_now

        if cam.prev_points.shape != points_now.shape:
            cam.prev_points = points_now.copy()
            return points_now

        delta = np.linalg.norm(points_now - cam.prev_points, axis=1)
        global_jump = float(np.median(delta))

        if global_jump > max_global_jump_m:
            return cam.prev_points.copy()

        filtered = alpha * cam.prev_points + (1.0 - alpha) * points_now

        jump_mask = delta > max_point_jump_m
        filtered[jump_mask] = cam.prev_points[jump_mask]

        cam.prev_points = filtered.copy()
        return filtered

    def draw_result(self, frame_bgr, mask, skeleton, valid_xy, depths_m, cam_name):
        vis = frame_bgr.copy()

        overlay = vis.copy()
        overlay[mask > 0] = (
            0.65 * overlay[mask > 0] + 0.35 * np.array([0, 0, 255])
        ).astype(np.uint8)

        vis = overlay

        ys, xs = np.where(skeleton)

        for x, y in zip(xs, ys):
            cv.circle(vis, (int(x), int(y)), 1, (0, 255, 255), -1)

        for i, (xy, d) in enumerate(zip(valid_xy, depths_m)):
            x = int(round(float(xy[0])))
            y = int(round(float(xy[1])))

            cv.circle(vis, (x, y), 5, (0, 255, 0), -1)
            cv.putText(
                vis,
                f"{cam_name}:{i} {d:.3f}m",
                (x + 6, y - 6),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        return vis

    def copy_header_with_frame(self, header, frame_id):
        out = type(header)()
        out.stamp = header.stamp
        out.frame_id = frame_id
        return out

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
        msg = self.bridge.cv2_to_imgmsg(mask_u8.astype(np.uint8), encoding="mono8")
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
    node = SAM31TwoCameraRopeDepthNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()