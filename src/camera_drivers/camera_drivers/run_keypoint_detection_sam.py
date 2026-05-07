#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import torch
import rclpy
import json
import sys

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from PIL import Image as PILImage

from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, CompressedImage
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import sensor_msgs_py.point_cloud2 as pc2

from skimage.morphology import skeletonize

for workspace_root in [Path.cwd(), *Path(__file__).resolve().parents]:
    for src_dir in (workspace_root, workspace_root / "src"):
        for local_package in ("sam3", "tapnet"):
            local_package_path = src_dir / local_package
            if local_package_path.exists() and str(local_package_path) not in sys.path:
                sys.path.insert(0, str(local_package_path))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from tapnet.tapnext.tapnext_torch import TAPNext


CKPT_SIZE = (256, 256)


def load_cam2_to_cam1(calib_path):
    with open(calib_path, "r") as f:
        calib = json.load(f)

    R = np.array(calib["camera_poses"]["cam2_to_cam1"]["R"], dtype=np.float64)
    T = np.array(calib["camera_poses"]["cam2_to_cam1"]["T"], dtype=np.float64)

    T_cam2_to_cam1 = np.eye(4, dtype=np.float64)
    T_cam2_to_cam1[:3, :3] = R
    T_cam2_to_cam1[:3, 3] = T

    return T_cam2_to_cam1


@dataclass
class CameraState:
    name: str
    image_topic: str
    depth_topic: str
    camera_info_topic: str

    fx: float = None
    fy: float = None
    cx: float = None
    cy: float = None

    initialized: bool = False
    tracking_state: object = None
    query_points_np_orig: np.ndarray = None


class TwoCameraTAPNextFusionNode(Node):
    def __init__(self):
        super().__init__("two_camera_tapnext_fusion_node")

        self.cam1 = CameraState(
            name="cam1",
            image_topic="/front_camera/color/image_raw",
            depth_topic="/front_camera/depth/image_raw",
            camera_info_topic="/front_camera/color/camera_info",
        )

        self.cam2 = CameraState(
            name="cam2",
            image_topic="/back_camera/color/image_raw",
            depth_topic="/back_camera/depth/image_raw",
            camera_info_topic="/back_camera/color/camera_info",
        )

        self.global_frame = "cam1_frame"

        self.pointcloud_topic = "/tapnext/fused_points_3d"
        self.cam1_pointcloud_topic = "/tapnext/cam1/points_3d"
        self.cam2_pointcloud_topic = "/tapnext/cam2/points_3d"
        self.annotated_topic_cam1 = "/tapnext/cam1/annotated_image/compressed"
        self.annotated_topic_cam2 = "/tapnext/cam2/annotated_image/compressed"

        self.fused_pub = self.create_publisher(PointCloud2, self.pointcloud_topic, 10)
        self.cam1_points_pub = self.create_publisher(PointCloud2, self.cam1_pointcloud_topic, 10)
        self.cam2_points_pub = self.create_publisher(PointCloud2, self.cam2_pointcloud_topic, 10)

        self.annotated_pub_cam1 = self.create_publisher(
            CompressedImage,
            self.annotated_topic_cam1,
            10,
        )
        self.annotated_pub_cam2 = self.create_publisher(
            CompressedImage,
            self.annotated_topic_cam2,
            10,
        )

        self.declare_parameter("cam2_tx", -0.25)
        self.declare_parameter("cam2_ty", 0.1)
        self.declare_parameter("cam2_tz", 1.44)
        self.declare_parameter("cam2_roll", 3.14)
        self.declare_parameter("cam2_pitch", -0.55)
        self.declare_parameter("cam2_yaw", -0.15)
        self.declare_parameter("use_calibration_base", True)
        self.declare_parameter(
            "sam_checkpoint_path",
            "/home/jeffreyfang/deformables/src/sam3/checkpoint/sam3.pt",
        )
        self.declare_parameter("sam_prompt", "rope")
        self.declare_parameter("sam_confidence_threshold", 0.35)
        self.declare_parameter("num_query_points", 20)
        self.declare_parameter("visualize_sam_keypoints", False)

        self.calib_path = "/home/jeffreyfang/calib/dataset/calibration.json"
        self.T_cam2_to_cam1_calib = load_cam2_to_cam1(self.calib_path)
        self.T_cam1_to_cam1 = np.eye(4, dtype=np.float64)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.get_logger().info(f"Using device: {self.device}")

        # -----------------------------
        # TAPNext
        # -----------------------------
        self.ckpt_path = "/home/jeffreyfang/deformables/src/tapnet/checkpoints/tapnextpp_ckpt.pt"

        self.model = TAPNext(image_size=CKPT_SIZE)
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        self.model.load_state_dict(
            {k.replace("tapnext.", ""): v for k, v in ckpt["state_dict"].items()}
        )
        self.model = self.model.to(self.device).eval()

        # -----------------------------
        # SAM image model
        # -----------------------------
        self.get_logger().info("Loading SAM image processor...")

        self.sam_checkpoint_path = str(self.get_parameter("sam_checkpoint_path").value)
        self.sam_prompt = str(self.get_parameter("sam_prompt").value)
        self.sam_confidence_threshold = float(
            self.get_parameter("sam_confidence_threshold").value
        )
        self.num_query_points = int(self.get_parameter("num_query_points").value)
        self.visualize_sam_keypoints = bool(
            self.get_parameter("visualize_sam_keypoints").value
        )

        sam_model = build_sam3_image_model(
            checkpoint_path=self.sam_checkpoint_path,
            device=str(self.device),
        )

        self.sam_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        sam_model = sam_model.eval()

        self.sam_processor = Sam3Processor(
            sam_model,
            device=str(self.device),
            confidence_threshold=self.sam_confidence_threshold,
        )

        self.bridge = CvBridge()

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

        self.get_logger().info("Two-camera SAM + TAPNext fusion node started.")
        self.get_logger().info(f"Publishing cam1 cloud to {self.cam1_pointcloud_topic}")
        self.get_logger().info(f"Publishing cam2 cloud to {self.cam2_pointcloud_topic}")
        self.get_logger().info(f"Publishing combined cloud to {self.pointcloud_topic}")
        self.get_logger().info(f"Global frame: {self.global_frame}")

    def camera_info_callback(self, msg: CameraInfo, cam: CameraState):
        cam.fx = float(msg.k[0])
        cam.fy = float(msg.k[4])
        cam.cx = float(msg.k[2])
        cam.cy = float(msg.k[5])

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

    def crop_and_downscale(self, img_rgb):
        h, w = img_rgb.shape[:2]

        if w >= h:
            startx = w // 2 - h // 2
            starty = 0
            crop_size = h
        else:
            startx = 0
            starty = h // 2 - w // 2
            crop_size = w

        cropped = img_rgb[starty:starty + crop_size, startx:startx + crop_size]

        resized = cv.resize(
            cropped,
            (CKPT_SIZE[1], CKPT_SIZE[0]),
            interpolation=cv.INTER_AREA,
        )

        return cropped, resized, (startx, starty, crop_size)

    def crop_depth_with_rgb_crop(self, depth_img, crop_info):
        startx, starty, crop_size = crop_info
        return depth_img[starty:starty + crop_size, startx:startx + crop_size]

    def preprocess_frame(self, frame_rgb):
        cropped_rgb, resized_rgb, crop_info = self.crop_and_downscale(frame_rgb)

        resized_rgb = cv.GaussianBlur(resized_rgb, (5, 5), 0)

        lab = cv.cvtColor(resized_rgb, cv.COLOR_RGB2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        resized_rgb = cv.merge((l, a, b))
        resized_rgb = cv.cvtColor(resized_rgb, cv.COLOR_LAB2RGB)

        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )
        resized_rgb = cv.filter2D(resized_rgb, -1, kernel)

        video_np = resized_rgb[None, None].astype(np.float32)
        video = torch.from_numpy(video_np)
        video = (video / 255.0) * 2.0 - 1.0
        video = video.to(self.device)

        return cropped_rgb, resized_rgb, video, crop_info

    def select_query_points_sam(self, cropped_rgb, cam_name):
        try:
            image = PILImage.fromarray(cropped_rgb)

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
                self.get_logger().warn(f"{cam_name}: SAM returned no masks")
                return None

            if torch.is_tensor(masks):
                masks_np = masks.detach().to(dtype=torch.bool).cpu().numpy()
            else:
                masks_np = np.asarray(masks)

            if torch.is_tensor(scores):
                scores_np = scores.detach().to(dtype=torch.float32).cpu().numpy()
            elif scores is None:
                scores_np = None
            else:
                scores_np = np.asarray(scores)

            if masks_np.shape[0] == 0:
                self.get_logger().warn(f"{cam_name}: SAM returned no masks")
                return None

            if scores_np is not None and scores_np.shape[0] == masks_np.shape[0]:
                mask_idx = int(np.argmax(scores_np))
            else:
                flattened = masks_np.reshape(masks_np.shape[0], -1)
                mask_idx = int(np.argmax(np.sum(flattened > 0, axis=1)))

            mask = masks_np[mask_idx]

            if torch.is_tensor(mask):
                mask = mask.detach().cpu().numpy()

            mask = np.asarray(mask)

            if mask.ndim > 2:
                mask = np.squeeze(mask)

            binary_mask = mask > 0.0

            if np.sum(binary_mask) < 20:
                self.get_logger().warn(f"{cam_name}: SAM rope mask too small")
                return None

            skeleton = skeletonize(binary_mask)

            ordered = self.order_skeleton_points_graph(skeleton)

            if ordered.shape[0] < 2:
                self.get_logger().warn(f"{cam_name}: ordered skeleton too small")
                return None

            if ordered.shape[0] <= self.num_query_points:
                keypoints_xy = ordered
            else:
                idx = np.linspace(
                    0,
                    ordered.shape[0] - 1,
                    self.num_query_points,
                ).astype(np.int32)
                keypoints_xy = ordered[idx]

            query_points = np.zeros((keypoints_xy.shape[0], 3), dtype=np.float32)
            query_points[:, 0] = 0.0
            query_points[:, 1] = keypoints_xy[:, 1]
            query_points[:, 2] = keypoints_xy[:, 0]

            if self.visualize_sam_keypoints:
                self.visualize_selected_sam_keypoints(
                    cropped_rgb=cropped_rgb,
                    binary_mask=binary_mask,
                    skeleton=skeleton,
                    keypoints_xy=keypoints_xy,
                    cam_name=cam_name,
                )

            self.get_logger().info(
                f"{cam_name}: SAM selected {query_points.shape[0]} query points"
            )

            return query_points

        except Exception as e:
            self.get_logger().error(f"{cam_name}: SAM keypoint selection failed: {e}")
            return None

    def order_skeleton_points_graph(self, skeleton):
        ys, xs = np.where(skeleton)

        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float32)

        nodes = set((int(x), int(y)) for x, y in zip(xs, ys))

        if len(nodes) <= 2:
            return np.asarray(list(nodes), dtype=np.float32)

        def get_neighbors(p, valid_nodes):
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

                for nb in get_neighbors(p, nodes):
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)

            components.append(comp)

        largest_component = max(components, key=len)
        nodes = set(largest_component)

        degrees = {
            p: len(get_neighbors(p, nodes))
            for p in nodes
        }

        endpoints = [p for p, d in degrees.items() if d == 1]

        def bfs_farthest(start):
            q = deque([start])
            parent = {start: None}
            dist = {start: 0}
            farthest = start

            while q:
                p = q.popleft()

                if dist[p] > dist[farthest]:
                    farthest = p

                for nb in get_neighbors(p, nodes):
                    if nb not in parent:
                        parent[nb] = p
                        dist[nb] = dist[p] + 1
                        q.append(nb)

            return farthest, parent, dist

        if len(endpoints) >= 2:
            start = endpoints[0]
        else:
            start = next(iter(nodes))

        a, _, _ = bfs_farthest(start)
        b, parent, _ = bfs_farthest(a)

        path = []
        cur = b

        while cur is not None:
            path.append(cur)
            cur = parent[cur]

        path = path[::-1]

        return np.asarray(path, dtype=np.float32)

    def visualize_selected_sam_keypoints(
        self,
        cropped_rgb,
        binary_mask,
        skeleton,
        keypoints_xy,
        cam_name,
    ):
        vis = cropped_rgb.copy()

        mask_overlay = np.zeros_like(vis)
        mask_overlay[:, :, 2] = (binary_mask.astype(np.uint8) * 255)

        vis = cv.addWeighted(vis, 0.75, mask_overlay, 0.25, 0.0)

        ys, xs = np.where(skeleton)
        for x, y in zip(xs, ys):
            cv.circle(vis, (int(x), int(y)), 1, (0, 255, 255), -1)

        for i, (x, y) in enumerate(keypoints_xy.astype(np.int32)):
            cv.circle(vis, (int(x), int(y)), 5, (255, 0, 0), -1)
            cv.putText(
                vis,
                str(i),
                (int(x) + 6, int(y) - 6),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        cv.imshow(
            f"SAM selected keypoints: {cam_name}",
            cv.cvtColor(vis, cv.COLOR_RGB2BGR),
        )
        cv.waitKey(1)

    def initialize_tracker(self, cam: CameraState, video, crop_h):
        model_h, model_w = CKPT_SIZE

        scale_x = model_w / crop_h
        scale_y = model_h / crop_h

        query_points_np = cam.query_points_np_orig.copy()
        query_points_np[:, 1] *= scale_y
        query_points_np[:, 2] *= scale_x
        query_points_np = query_points_np[None]

        query_points = torch.from_numpy(query_points_np).to(self.device)

        with torch.no_grad():
            pred_tracks, _, visible_logits, cam.tracking_state = self.model(
                video=video,
                query_points=query_points,
            )

        tracks = pred_tracks.cpu().numpy()[0, 0]
        visible = (visible_logits > 0).cpu().numpy()[0, 0, :, 0]

        cam.initialized = True

        self.get_logger().info(
            f"Initialized {cam.name} TAPNext with {query_points_np.shape[1]} points"
        )

        return tracks, visible

    def step_tracker(self, cam: CameraState, video):
        with torch.no_grad():
            pred_tracks, _, visible_logits, cam.tracking_state = self.model(
                video=video,
                state=cam.tracking_state,
            )

        tracks = pred_tracks.cpu().numpy()[0, 0]
        visible = (visible_logits > 0).cpu().numpy()[0, 0, :, 0]

        return tracks, visible

    def query_depths_at_tracks(self, cropped_depth, tracks_xy, visible):
        h, w = cropped_depth.shape[:2]
        depths_m = []

        for xy, is_visible in zip(tracks_xy, visible):
            if not is_visible:
                depths_m.append(np.nan)
                continue

            x = int(round(float(xy[0])))
            y = int(round(float(xy[1])))

            if x < 0 or x >= w or y < 0 or y >= h:
                depths_m.append(np.nan)
                continue

            x0 = max(0, x - 1)
            x1 = min(w, x + 2)
            y0 = max(0, y - 1)
            y1 = min(h, y + 2)

            patch = cropped_depth[y0:y1, x0:x1]

            if patch.size == 0:
                depths_m.append(np.nan)
                continue

            if cropped_depth.dtype == np.uint16:
                patch_m = patch.astype(np.float32) * 0.001
            else:
                patch_m = patch.astype(np.float32)

            valid = np.isfinite(patch_m) & (patch_m > 0.0)

            if not np.any(valid):
                depths_m.append(np.nan)
                continue

            d_m = float(np.min(patch_m[valid]))
            depths_m.append(d_m)

        return np.asarray(depths_m, dtype=np.float32)

    def tracks_to_camera_points(self, cam: CameraState, tracks_xy, visible, depths_m, crop_info):
        if cam.fx is None:
            self.get_logger().warn(f"{cam.name} intrinsics not received yet.")
            return np.empty((0, 3), dtype=np.float32), []

        startx, starty, _ = crop_info
        points = []
        ids = []

        for i, (xy, is_visible, z) in enumerate(zip(tracks_xy, visible, depths_m)):
            if not is_visible:
                continue
            if not np.isfinite(z) or z <= 0.0:
                continue

            u = float(xy[0]) + float(startx)
            v = float(xy[1]) + float(starty)

            x_c = (u - cam.cx) * z / cam.fx
            y_c = (v - cam.cy) * z / cam.fy
            z_c = float(z)

            points.append([x_c, y_c, z_c])
            ids.append(i)

        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float32), []

        return np.asarray(points, dtype=np.float32), ids

    def transform_points(self, points, T):
        if points.shape[0] == 0:
            return points

        ones = np.ones((points.shape[0], 1), dtype=np.float64)
        points_h = np.concatenate([points.astype(np.float64), ones], axis=1)
        out_h = (T @ points_h.T).T
        return out_h[:, :3].astype(np.float32)

    def process_camera(self, cam: CameraState, rgb_msg, depth_msg):
        frame_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)

        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth_img = np.asarray(depth_img)

        cropped_rgb, resized_rgb, video, crop_info = self.preprocess_frame(frame_rgb)
        cropped_depth = self.crop_depth_with_rgb_crop(depth_img, crop_info)

        crop_h, crop_w = cropped_rgb.shape[:2]

        if not cam.initialized:
            self.get_logger().info(
                f"{cam.name}: first frame. Selecting query points with SAM."
            )

            qp = self.select_query_points_sam(cropped_rgb, cam.name)

            if qp is None:
                self.get_logger().warn(f"{cam.name}: no SAM query points selected.")
                return None

            cam.query_points_np_orig = qp
            tracks_yx, visible = self.initialize_tracker(cam, video, crop_h)
        else:
            tracks_yx, visible = self.step_tracker(cam, video)

        tracks_xy = tracks_yx[:, ::-1]

        model_h, model_w = CKPT_SIZE
        tracks_xy[:, 0] *= crop_w / model_w
        tracks_xy[:, 1] *= crop_h / model_h

        depths_m = self.query_depths_at_tracks(cropped_depth, tracks_xy, visible)

        points_cam, ids = self.tracks_to_camera_points(
            cam,
            tracks_xy,
            visible,
            depths_m,
            crop_info,
        )

        annotated_rgb = self.draw_tracks_with_depth(
            cropped_rgb,
            tracks_xy,
            visible,
            depths_m,
            cam.name,
        )

        return {
            "points_cam": points_cam,
            "ids": ids,
            "tracks_xy": tracks_xy,
            "visible": visible,
            "depths_m": depths_m,
            "annotated_rgb": annotated_rgb,
        }

    def draw_tracks_with_depth(self, cropped_rgb, tracks_xy, visible, depths_m, cam_name):
        vis_img = cropped_rgb.copy()

        for i, (xy, is_visible, d_m) in enumerate(zip(tracks_xy, visible, depths_m)):
            if not is_visible:
                continue

            x = int(round(float(xy[0])))
            y = int(round(float(xy[1])))

            cv.circle(vis_img, (x, y), 5, (0, 255, 0), -1)

            if np.isfinite(d_m):
                label = f"{cam_name}:{i} {d_m:.3f}m"
            else:
                label = f"{cam_name}:{i} no depth"

            cv.putText(
                vis_img,
                label,
                (x + 6, y - 6),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        return vis_img

    def publish_annotated(self, pub, rgb_img, header):
        bgr = cv.cvtColor(rgb_img, cv.COLOR_RGB2BGR)
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
        pub.publish(msg)

    def synced_callback(self, cam1_rgb_msg, cam1_depth_msg, cam2_rgb_msg, cam2_depth_msg):
        try:
            result1 = self.process_camera(self.cam1, cam1_rgb_msg, cam1_depth_msg)
            result2 = self.process_camera(self.cam2, cam2_rgb_msg, cam2_depth_msg)

            if result1 is None or result2 is None:
                return

            cam1_points_global = self.transform_points(
                result1["points_cam"],
                self.T_cam1_to_cam1,
            )

            T_cam2_live = self.get_cam2_transform_live()

            cam2_points_global = self.transform_points(
                result2["points_cam"],
                T_cam2_live,
            )

            header = cam1_rgb_msg.header
            header.frame_id = self.global_frame

            cam1_cloud_msg = pc2.create_cloud_xyz32(
                header,
                cam1_points_global.tolist(),
            )

            cam2_cloud_msg = pc2.create_cloud_xyz32(
                header,
                cam2_points_global.tolist(),
            )

            self.cam1_points_pub.publish(cam1_cloud_msg)
            self.cam2_points_pub.publish(cam2_cloud_msg)

            if cam1_points_global.shape[0] == 0 and cam2_points_global.shape[0] == 0:
                fused_points = np.empty((0, 3), dtype=np.float32)
            elif cam1_points_global.shape[0] == 0:
                fused_points = cam2_points_global
            elif cam2_points_global.shape[0] == 0:
                fused_points = cam1_points_global
            else:
                fused_points = np.concatenate(
                    [cam1_points_global, cam2_points_global],
                    axis=0,
                )

            cloud_msg = pc2.create_cloud_xyz32(
                header,
                fused_points.tolist(),
            )

            self.fused_pub.publish(cloud_msg)

            self.publish_annotated(
                self.annotated_pub_cam1,
                result1["annotated_rgb"],
                cam1_rgb_msg.header,
            )

            self.publish_annotated(
                self.annotated_pub_cam2,
                result2["annotated_rgb"],
                cam2_rgb_msg.header,
            )

            self.get_logger().info(
                f"cam1 pts={cam1_points_global.shape[0]}, "
                f"cam2 pts={cam2_points_global.shape[0]}, "
                f"combined pts={fused_points.shape[0]}"
            )

        except Exception as e:
            self.get_logger().error(f"Synced callback failed: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = TwoCameraTAPNextFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    cv.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
