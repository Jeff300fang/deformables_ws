#!/usr/bin/env python3

import sys
from pathlib import Path
from contextlib import nullcontext

import cv2 as cv
import numpy as np
import torch
import rclpy
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from PIL import Image as PILImage
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Bool
from rcl_interfaces.msg import SetParametersResult
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped

for workspace_root in [Path.cwd(), *Path(__file__).resolve().parents]:
    for src_dir in (workspace_root, workspace_root / "src"):
        local_package_path = src_dir / "sam3"
        if local_package_path.exists() and str(local_package_path) not in sys.path:
            sys.path.insert(0, str(local_package_path))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SingleCameraSAMClothGridNode(Node):
    def __init__(self):
        super().__init__("front_single_camera_sam_cloth_grid_node")

        self.declare_parameter(
            "sam_checkpoint_path",
            "/home/jeffreyfang/deformables_ws/src/perception/checkpoints/sam3.pt",
        )
        self.declare_parameter("sam_prompt", "cloth")
        self.declare_parameter("sam_confidence_threshold", 0.05)
        self.declare_parameter("run_sam_every_frame", True)
        self.declare_parameter("resize_width", 384)
        self.declare_parameter("target_hz", 10.0)

        self.declare_parameter("rgb_topic", "/front_camera/color/image_raw")
        self.declare_parameter("depth_topic", "/front_camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/front_camera/color/camera_info")

        self.declare_parameter("mask_topic", "/front_camera/sam_cloth/mask")
        self.declare_parameter(
            "annotated_topic",
            "/front_camera/sam_cloth/annotated_image/compressed",
        )

        self.declare_parameter(
            "observed_points_topic",
            "/front/sam_cloth/observed_points_3d",
        )
        self.declare_parameter(
            "cloth_grid_cloud_topic",
            "/front/sam_cloth/cloth_grid_cloud",
        )
        self.declare_parameter(
            "cloth_grid_pose_topic",
            "/front/sam_cloth/cloth_grid_poses",
        )

        self.declare_parameter("output_frame_id", "workstation")

        self.declare_parameter("depth_search_radius", 4)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 5.0)

        # Simulation cloth grid params
        self.declare_parameter("cloth_grid_n", 10)
        self.declare_parameter("cloth_spacing_m", 0.04233)
        self.declare_parameter("table_z", 0.0)

        # Dense image sampling params
        self.declare_parameter("mask_sample_stride_px", 4)
        self.declare_parameter("max_observed_points", 3000)

        # Translated frame params
        self.declare_parameter("body_translation_x", 0.85)
        self.declare_parameter("body_translation_y", 0.0)
        self.declare_parameter("body_translation_z", 0.29)

        self.declare_parameter("body_rotation_roll_deg", 0.0)
        self.declare_parameter("body_rotation_pitch_deg", 0.0)
        self.declare_parameter("body_rotation_yaw_deg", 0.0)

        self.sam_checkpoint_path = str(self.get_parameter("sam_checkpoint_path").value)
        self.sam_prompt = str(self.get_parameter("sam_prompt").value)
        self.sam_confidence_threshold = float(
            self.get_parameter("sam_confidence_threshold").value
        )
        self.run_sam_every_frame = bool(
            self.get_parameter("run_sam_every_frame").value
        )
        self.resize_width = int(self.get_parameter("resize_width").value)
        self.target_hz = float(self.get_parameter("target_hz").value)

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)

        self.mask_topic = str(self.get_parameter("mask_topic").value)
        self.annotated_topic = str(self.get_parameter("annotated_topic").value)
        self.observed_points_topic = str(self.get_parameter("observed_points_topic").value)
        self.cloth_grid_cloud_topic = str(
            self.get_parameter("cloth_grid_cloud_topic").value
        )
        self.cloth_grid_pose_topic = str(
            self.get_parameter("cloth_grid_pose_topic").value
        )

        self.output_frame_id = str(self.get_parameter("output_frame_id").value)

        self.depth_search_radius = int(
            self.get_parameter("depth_search_radius").value
        )
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        self.cloth_grid_n = int(self.get_parameter("cloth_grid_n").value)
        self.cloth_spacing_m = float(self.get_parameter("cloth_spacing_m").value)
        self.table_z = float(self.get_parameter("table_z").value)

        self.mask_sample_stride_px = int(
            self.get_parameter("mask_sample_stride_px").value
        )
        self.max_observed_points = int(
            self.get_parameter("max_observed_points").value
        )

        self.body_translation_x = float(self.get_parameter("body_translation_x").value)
        self.body_translation_y = float(self.get_parameter("body_translation_y").value)
        self.body_translation_z = float(self.get_parameter("body_translation_z").value)

        self.body_rotation_roll_deg = float(self.get_parameter("body_rotation_roll_deg").value)
        self.body_rotation_pitch_deg = float(self.get_parameter("body_rotation_pitch_deg").value)
        self.body_rotation_yaw_deg = float(self.get_parameter("body_rotation_yaw_deg").value)

        self.add_on_set_parameters_callback(self.parameter_callback)

        self.bridge = CvBridge()

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.last_mask = None
        self.latest_mask = None
        self.last_process_time = self.get_clock().now()
        self.stop = False

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.sam_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

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

        self.info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            1,
        )

        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            1,
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            1,
        )

        self.stop_front_sub = self.create_subscription(
            Bool,
            "/stop_front",
            self.stop_callback,
            1,
        )

        self.mask_pub = self.create_publisher(Image, self.mask_topic, 1)

        self.annotated_pub = self.create_publisher(
            CompressedImage,
            self.annotated_topic,
            1,
        )

        self.observed_points_pub = self.create_publisher(
            PointCloud2,
            self.observed_points_topic,
            1,
        )

        self.grid_cloud_pub = self.create_publisher(
            PointCloud2,
            self.cloth_grid_cloud_topic,
            1,
        )

        self.grid_pose_pub = self.create_publisher(
            PoseArray,
            self.cloth_grid_pose_topic,
            1,
        )

        self.get_logger().info(f"Subscribed RGB: {self.rgb_topic}")
        self.get_logger().info(f"Subscribed depth: {self.depth_topic}")
        self.get_logger().info(f"Subscribed camera info: {self.camera_info_topic}")
        self.get_logger().info(f"Publishing observed points: {self.observed_points_topic}")
        self.get_logger().info(f"Publishing cloth grid cloud: {self.cloth_grid_cloud_topic}")
        self.get_logger().info(f"Publishing cloth grid poses: {self.cloth_grid_pose_topic}")
        self.left_ee_position = None
        self.right_ee_position = None
        self.left_ee_sub = self.create_subscription(
            PoseStamped,
            '/left/workstation/end_effector_pose',
            self.left_ee_pose_callback,
            1,
        )

        self.right_ee_sub = self.create_subscription(
            PoseStamped,
            '/right/workstation/end_effector_pose',
            self.right_ee_pose_callback,
            1,
        )


        self.cloth_direction_pub = self.create_publisher(
            MarkerArray,
            "/cloth_grid/direction_markers",
            10,
        )

        self.spline_pub = self.create_publisher(
            Marker,
            "/cloth_grid/fitted_spline",
            10,
        )

    def publish_spline_marker(self, dense_xyz):
        dense_xyz = np.asarray(dense_xyz, dtype=np.float32)

        if dense_xyz.ndim != 2 or dense_xyz.shape[1] != 3:
            return

        dense_xyz = dense_xyz[np.all(np.isfinite(dense_xyz), axis=1)]

        if dense_xyz.shape[0] < 2:
            return

        marker = Marker()
        marker.header.frame_id = self.output_frame_id
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "cloth_fitted_spline"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.01  # line thickness in meters

        marker.color.r = 0.0
        marker.color.g = 0.3
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.points = [
            Point(
                x=float(p[0]),
                y=float(p[1]),
                z=float(p[2]),
            )
            for p in dense_xyz
        ]

        marker.lifetime.sec = 0

        self.spline_pub.publish(marker)

    def left_ee_pose_callback(self, msg):
        self.left_ee_position = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float32,
        )


    def right_ee_pose_callback(self, msg):
        self.right_ee_position = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float32,
        )

    def stop_callback(self, msg):
        self.stop = bool(msg.data)

    def parameter_callback(self, params):
        try:
            for param in params:
                if param.name == "depth_search_radius":
                    self.depth_search_radius = int(param.value)
                elif param.name == "min_depth_m":
                    self.min_depth_m = float(param.value)
                elif param.name == "max_depth_m":
                    self.max_depth_m = float(param.value)
                elif param.name == "cloth_grid_n":
                    self.cloth_grid_n = int(param.value)
                elif param.name == "cloth_spacing_m":
                    self.cloth_spacing_m = float(param.value)
                elif param.name == "table_z":
                    self.table_z = float(param.value)
                elif param.name == "mask_sample_stride_px":
                    self.mask_sample_stride_px = int(param.value)
                elif param.name == "max_observed_points":
                    self.max_observed_points = int(param.value)
                elif param.name == "body_translation_x":
                    self.body_translation_x = float(param.value)
                elif param.name == "body_translation_y":
                    self.body_translation_y = float(param.value)
                elif param.name == "body_translation_z":
                    self.body_translation_z = float(param.value)
                elif param.name == "output_frame_id":
                    self.output_frame_id = str(param.value)
                elif param.name == "body_rotation_roll_deg":
                    self.body_rotation_roll_deg = float(param.value)
                elif param.name == "body_rotation_pitch_deg":
                    self.body_rotation_pitch_deg = float(param.value)
                elif param.name == "body_rotation_yaw_deg":
                    self.body_rotation_yaw_deg = float(param.value)

            return SetParametersResult(successful=True)
        except Exception as e:
            return SetParametersResult(successful=False, reason=str(e))

    def euler_rotation_matrix_deg(self, roll_deg, pitch_deg, yaw_deg):
        r = np.deg2rad(float(roll_deg))
        p = np.deg2rad(float(pitch_deg))
        y = np.deg2rad(float(yaw_deg))

        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)

        Rx = np.array([[1, 0, 0],
                    [0, cr, -sr],
                    [0, sr, cr]], dtype=np.float32)

        Ry = np.array([[cp, 0, sp],
                    [0, 1, 0],
                    [-sp, 0, cp]], dtype=np.float32)

        Rz = np.array([[cy, -sy, 0],
                    [sy, cy, 0],
                    [0, 0, 1]], dtype=np.float32)

        # yaw-pitch-roll / ZYX convention
        return Rz @ Ry @ Rx

    def camera_info_callback(self, msg):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

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

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

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

    def sample_mask_pixels(self, mask):
        stride = max(1, int(self.mask_sample_stride_px))

        ys, xs = np.nonzero(mask > 0)

        if len(xs) == 0:
            return None

        keep = (xs % stride == 0) & (ys % stride == 0)
        xs = xs[keep]
        ys = ys[keep]

        if len(xs) == 0:
            ys, xs = np.nonzero(mask > 0)

        pixels = np.stack([xs, ys], axis=1).astype(np.float32)

        if len(pixels) > self.max_observed_points:
            idx = np.random.choice(
                len(pixels),
                self.max_observed_points,
                replace=False,
            )
            pixels = pixels[idx]

        return pixels

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

    def rgb_callback(self, msg):
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
                    self.get_logger().warn("SAM found no cloth mask")
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
            self.latest_mask = mask_full

            mask_msg = self.bridge.cv2_to_imgmsg(
                (mask_full * 255).astype(np.uint8),
                encoding="mono8",
            )
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)

            annotated_rgb = self.draw_mask_overlay(frame_rgb, mask_full)
            self.publish_annotated(annotated_rgb, msg.header)

            self.get_logger().info(
                f"Published cloth mask. pixels={int(np.sum(mask_full > 0))}"
            )

        except Exception as e:
            self.get_logger().error(f"RGB callback failed: {e}")

    def depth_to_meters(self, depth_value, encoding):
        if encoding in ("16UC1", "mono16"):
            return float(depth_value) * 0.001

        if encoding == "32FC1":
            return float(depth_value)

        return float(depth_value)

    def get_valid_depth(self, depth_img, x, y, encoding):
        h, w = depth_img.shape[:2]

        x = int(round(float(x)))
        y = int(round(float(y)))

        if x < 0 or x >= w or y < 0 or y >= h:
            return None

        candidates = []
        r = self.depth_search_radius

        for yy in range(max(0, y - r), min(h, y + r + 1)):
            for xx in range(max(0, x - r), min(w, x + r + 1)):
                z = self.depth_to_meters(depth_img[yy, xx], encoding)

                if not np.isfinite(z):
                    continue

                if z < self.min_depth_m or z > self.max_depth_m:
                    continue

                dist2 = (xx - x) ** 2 + (yy - y) ** 2
                candidates.append((dist2, z))

        if len(candidates) == 0:
            return None

        candidates.sort(key=lambda p: p[0])
        nearby_depths = [z for _, z in candidates[:8]]

        return float(np.median(nearby_depths))

    def pixel_to_3d(self, x, y, z):
        X = (float(x) - self.cx) * z / self.fx
        Y = (float(y) - self.cy) * z / self.fy
        Z = z

        return np.array([X, Y, Z], dtype=np.float32)

    def optical_to_body_frame(self, points_3d):
        points_3d = np.asarray(points_3d, dtype=np.float32)

        if points_3d.shape[0] == 0:
            return points_3d

        x_opt = points_3d[:, 0]
        y_opt = points_3d[:, 1]
        z_opt = points_3d[:, 2]

        points_body = np.zeros_like(points_3d)

        points_body[:, 0] = z_opt
        points_body[:, 1] = -x_opt
        points_body[:, 2] = -y_opt

        points_body[:, 0] *= -1.0
        points_body[:, 1] *= -1.0

        points_body[:, 0] += self.body_translation_x
        points_body[:, 1] += self.body_translation_y
        points_body[:, 2] += self.body_translation_z

        R = self.euler_rotation_matrix_deg(
            self.body_rotation_roll_deg,
            self.body_rotation_pitch_deg,
            self.body_rotation_yaw_deg,
        )

        points_body = points_body @ R.T

        return points_body

    def observed_points_from_depth(self, depth_img, depth_encoding):
        if self.latest_mask is None:
            return None

        pixels = self.sample_mask_pixels(self.latest_mask)

        if pixels is None or len(pixels) == 0:
            return None

        points_3d = []

        for x, y in pixels:
            z = self.get_valid_depth(
                depth_img,
                x,
                y,
                depth_encoding,
            )

            if z is None:
                continue

            p3d = self.pixel_to_3d(x, y, z)
            points_3d.append(p3d)

        if len(points_3d) == 0:
            return None

        points_3d = np.asarray(points_3d, dtype=np.float32)
        points_3d = self.optical_to_body_frame(points_3d)

        return points_3d

    def publish_cloth_direction_markers(self, grid):
        marker_array = MarkerArray()

        for i in range(grid.shape[0] - 1):
            p0 = np.mean(grid[i], axis=0)
            p1 = np.mean(grid[i + 1], axis=0)

            marker = Marker()
            marker.header.frame_id = "workstation"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "cloth_direction"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            marker.points = [
                Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
                Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])),
            ]

            marker.scale.x = 0.01
            marker.scale.y = 0.025
            marker.scale.z = 0.04

            marker.color.r = 1.0
            marker.color.g = 0.2
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker.lifetime.sec = 0
            marker_array.markers.append(marker)

        self.cloth_direction_pub.publish(marker_array)

    def build_cloth_grid_from_points(self, points_3d):
        points = np.asarray(points_3d, dtype=np.float32)
        points = points[np.all(np.isfinite(points), axis=1)]

        if points.shape[0] < 30:
            return None

        n = int(self.cloth_grid_n)
        spacing = float(self.cloth_spacing_m)

        if n < 2 or spacing <= 0.0:
            return None

        table_z = float(self.table_z)

        def remove_duplicate_xz(path_xz):
            if path_xz is None or len(path_xz) < 2:
                return path_xz
            seg = np.diff(path_xz, axis=0)
            keep = np.linalg.norm(seg, axis=1) > 1e-6
            return np.vstack([path_xz[0], path_xz[1:][keep]]).astype(np.float32)

        def path_length(path_xz):
            if path_xz is None or len(path_xz) < 2:
                return 0.0
            return float(np.sum(np.linalg.norm(np.diff(path_xz, axis=0), axis=1)))

        def resample_path(path_xz, row_s):
            path_xz = remove_duplicate_xz(path_xz)
            if path_xz is None or len(path_xz) < 2:
                return None

            seg = np.diff(path_xz, axis=0)
            seg_len = np.linalg.norm(seg, axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg_len)])
            total_len = float(s[-1])

            row_xz = np.zeros((len(row_s), 2), dtype=np.float32)

            for i, target_s in enumerate(row_s):
                if target_s <= total_len:
                    idx = int(np.searchsorted(s, target_s, side="right") - 1)
                    idx = max(0, min(idx, len(seg_len) - 1))
                    alpha = float((target_s - s[idx]) / max(seg_len[idx], 1e-6))
                    row_xz[i] = (1.0 - alpha) * path_xz[idx] + alpha * path_xz[idx + 1]
                else:
                    row_xz[i] = path_xz[-1]

            return row_xz

        def fit_table_path(table_points):
            if table_points.shape[0] < 5:
                return None

            x = table_points[:, 0]
            z = table_points[:, 2]

            x_lo, x_hi = np.percentile(x, [2.0, 98.0])
            num_bins = max(8, min(40, table_points.shape[0] // 15))
            bins = np.linspace(x_hi, x_lo, num_bins + 1)

            path = []

            for i in range(num_bins):
                hi = bins[i]
                lo = bins[i + 1]
                mask = (x <= hi) & (x >= lo) if i == num_bins - 1 else (x <= hi) & (x > lo)

                if np.count_nonzero(mask) < 4:
                    continue

                path.append([float(np.median(x[mask])), float(np.median(z[mask]))])

            if len(path) < 2:
                return None

            path = np.asarray(path, dtype=np.float32)
            path[:, 1] = table_z

            if path[0, 0] < path[-1, 0]:
                path = path[::-1]

            return remove_duplicate_xz(path)

        def fit_upright_path(upright_points, table_end_xz, gripper_z=None):
            if upright_points.shape[0] < 5:
                return None

            x = upright_points[:, 0]
            z = upright_points[:, 2]

            z_lo = float(np.percentile(z, 2.0))
            z_hi = float(np.percentile(z, 98.0))

            if gripper_z is not None and np.isfinite(gripper_z):
                z_top = float(gripper_z)
                z_bottom = min(z_lo, float(table_end_xz[1]))
            else:
                z_top = z_hi
                z_bottom = z_lo

            if z_top < z_bottom:
                z_top, z_bottom = z_bottom, z_top

            if z_top - z_bottom < 1e-6:
                return None

            z_targets = list(np.arange(z_top, z_bottom, -spacing, dtype=np.float32))

            if len(z_targets) == 0 or abs(float(z_targets[-1]) - z_bottom) > 0.35 * spacing:
                z_targets.append(np.float32(z_bottom))

            path_top_down = []
            half_band = 0.5 * spacing
            min_points_per_band = 4

            for z_target in z_targets:
                z_target = float(z_target)

                mask = np.abs(z - z_target) <= half_band

                if np.count_nonzero(mask) < min_points_per_band:
                    mask = np.abs(z - z_target) <= spacing

                if np.count_nonzero(mask) >= min_points_per_band:
                    bx = float(np.median(x[mask]))
                else:
                    k = min(max(min_points_per_band, 8), upright_points.shape[0])
                    nearest = np.argsort(np.abs(z - z_target))[:k]
                    bx = float(np.median(x[nearest]))

                path_top_down.append([bx, z_target])

            if len(path_top_down) < 2:
                return None

            path = np.asarray(path_top_down[::-1], dtype=np.float32)

            if np.linalg.norm(path[0] - table_end_xz) < spacing:
                path[0] = table_end_xz

            return remove_duplicate_xz(path)

        z = points[:, 2]
        y = points[:, 1]

        table_band = max(0.05, 0.35 * spacing)
        table_mask = np.abs(z - table_z) <= table_band

        if np.count_nonzero(table_mask) < 10:
            low_z = np.percentile(z, 25.0)
            table_mask = z <= low_z

        table_points = points[table_mask]
        upright_points = points[~table_mask]

        if table_points.shape[0] < 5:
            return None

        table_path = fit_table_path(table_points)

        if table_path is None or len(table_path) < 2:
            return None

        table_end_xz = table_path[-1]

        have_ee = (
            self.left_ee_position is not None
            and self.right_ee_position is not None
        )

        gripper_z = None
        if have_ee:
            ee_mid = 0.5 * (self.left_ee_position + self.right_ee_position)
            gripper_z = float(ee_mid[2])

        upright_path = fit_upright_path(upright_points, table_end_xz, gripper_z)

        if upright_path is not None and len(upright_path) >= 2:
            if np.linalg.norm(upright_path[0] - table_path[-1]) < spacing:
                center_xz = np.vstack([table_path, upright_path[1:]])
            else:
                center_xz = np.vstack([table_path, upright_path])
        else:
            center_xz = table_path

        if have_ee:
            ee_mid = 0.5 * (self.left_ee_position + self.right_ee_position)
            ee_xz = np.array([ee_mid[0], ee_mid[2]], dtype=np.float32)

            if np.linalg.norm(ee_xz - center_xz[-1]) > 0.25 * spacing:
                center_xz = np.vstack([center_xz, ee_xz])

        center_xz = remove_duplicate_xz(center_xz)

        if center_xz is None or len(center_xz) < 2:
            return None

        flat_len = path_length(table_path)
        upright_len = path_length(upright_path) if upright_path is not None else 0.0

        if have_ee and upright_path is not None and len(upright_path) >= 2:
            ee_mid = 0.5 * (self.left_ee_position + self.right_ee_position)
            ee_xz = np.array([ee_mid[0], ee_mid[2]], dtype=np.float32)
            upright_len = max(upright_len, path_length(np.vstack([upright_path[0], ee_xz])))

        visible_len = flat_len + upright_len
        expected_total_len = spacing * float(n - 1)
        missing_len = max(0.0, expected_total_len - visible_len)
        folded_extension = 0.5 * missing_len

        def make_center_with_hidden(fold_len):
            if fold_len <= 1e-4:
                return center_xz.copy()

            bend_xz = table_path[-1].copy()
            bend_xz[1] = table_z

            table_dir = table_path[-1] - table_path[-2]
            table_norm = float(np.linalg.norm(table_dir))

            if table_norm < 1e-6:
                table_dir = np.array([-1.0, 0.0], dtype=np.float32)
            else:
                table_dir = table_dir / table_norm

            hidden_xz = bend_xz + table_dir * fold_len
            hidden_xz[1] = table_z

            return np.vstack(
                [
                    table_path,
                    hidden_xz[None, :],
                    center_xz[len(table_path):],
                ]
            )

        center_xz = make_center_with_hidden(folded_extension)

        if folded_extension > 1e-4:
            total_len = path_length(center_xz)
            excess_len = max(0.0, total_len - expected_total_len)

            if excess_len > 1e-4:
                folded_extension = max(0.0, folded_extension - 0.5 * excess_len)
                center_xz = make_center_with_hidden(folded_extension)

        center_xz = remove_duplicate_xz(center_xz)

        if center_xz is None or len(center_xz) < 2:
            return None

        if have_ee:
            ee_mid = 0.5 * (self.left_ee_position + self.right_ee_position)
            ee_xz = np.array([ee_mid[0], ee_mid[2]], dtype=np.float32)

            if np.linalg.norm(center_xz[-1] - ee_xz) > 1e-6:
                center_xz = np.vstack([center_xz, ee_xz])
                center_xz = remove_duplicate_xz(center_xz)

            total_len = path_length(center_xz)

            if total_len < 1e-6:
                return None

            row_s = np.linspace(0.0, total_len, n, dtype=np.float32)
            row_xz = resample_path(center_xz, row_s)

            if row_xz is not None:
                row_xz[0] = center_xz[0]
                row_xz[-1] = ee_xz
        else:
            row_s = np.arange(n, dtype=np.float32) * spacing
            row_xz = resample_path(center_xz, row_s)

        if row_xz is None:
            return None

        if have_ee:
            ee_mid = 0.5 * (self.left_ee_position + self.right_ee_position)
            y_center = float(ee_mid[1])

            y_sign = np.sign(self.right_ee_position[1] - self.left_ee_position[1])
            if abs(y_sign) < 1e-6:
                y_sign = 1.0

            width_dir = np.array([0.0, y_sign, 0.0], dtype=np.float32)
        else:
            y_center = float(np.median(y))
            width_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        col_offsets = (
            np.arange(n, dtype=np.float32) - 0.5 * float(n - 1)
        ) * spacing

        grid = np.zeros((n, n, 3), dtype=np.float32)

        for i in range(n):
            row_center = np.array(
                [row_xz[i, 0], y_center, row_xz[i, 1]],
                dtype=np.float32,
            )

            for j in range(n):
                grid[i, j] = row_center + width_dir * col_offsets[j]

        observed_mask = np.zeros((n, n), dtype=bool)
        radius2 = (0.8 * spacing) ** 2

        for i in range(n):
            for j in range(n):
                d = points - grid[i, j]
                observed_mask[i, j] = bool(np.any(np.sum(d * d, axis=1) <= radius2))


        self.publish_cloth_direction_markers(grid)

        return grid, observed_mask

    def fix_grid(self, grid_result, points_3d=None):
        """Rebuild the cloth grid by fixed Euclidean row spacing along a fitted centerline.

        Behavior:
        - Uses the input grid row centers as an ordered centerline.
        - Fits/samples a smooth centerline through those row centers.
        - If fixed-spacing sampling would put row 0 in front of the original first row,
        extra length is inserted at the spline bend/knee, not at the gripper endpoint.
        - The bend/knee is detected as the point of largest direction change.
        - The final gripper endpoint is not extended upward/past the grippers.
        """
        if grid_result is None:
            return None

        if isinstance(grid_result, tuple):
            grid = grid_result[0]
        else:
            grid = grid_result

        grid = np.asarray(grid, dtype=np.float32)

        if grid.ndim != 3 or grid.shape[2] != 3:
            return None

        spacing = float(self.cloth_spacing_m)

        if spacing <= 0.0 or grid.shape[0] < 2 or grid.shape[1] < 2:
            return None

        n_rows = int(grid.shape[0])
        n_cols = int(grid.shape[1])

        row_centers = np.mean(grid, axis=1).astype(np.float32)
        finite = np.all(np.isfinite(row_centers), axis=1)
        row_centers = row_centers[finite]

        if row_centers.shape[0] < 2:
            return None

        center_xz = row_centers[:, [0, 2]].astype(np.float32)
        center_y = row_centers[:, 1].astype(np.float32)

        seg = np.diff(center_xz, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        keep_seg = seg_len > 1e-6

        if not np.any(keep_seg):
            return None

        keep_pts = np.concatenate([[True], keep_seg])
        center_xz = center_xz[keep_pts]
        center_y = center_y[keep_pts]

        if center_xz.shape[0] < 2:
            return None

        seg = np.diff(center_xz, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg_len)]).astype(np.float32)
        polyline_len = float(s[-1])

        if polyline_len < 1e-6:
            return None

        def linear_eval(values, targets):
            values = np.asarray(values, dtype=np.float32)
            targets = np.asarray(targets, dtype=np.float32)
            out = np.zeros((len(targets), values.shape[1]), dtype=np.float32)

            for i, target in enumerate(targets):
                target = float(target)

                if target <= s[0]:
                    idx = 0
                elif target >= s[-1]:
                    idx = len(s) - 2
                else:
                    idx = int(np.searchsorted(s, target, side="right") - 1)
                    idx = max(0, min(idx, len(s) - 2))

                denom = max(float(s[idx + 1] - s[idx]), 1e-6)
                alpha = float((target - s[idx]) / denom)
                out[i] = (1.0 - alpha) * values[idx] + alpha * values[idx + 1]

            return out

        dense_count = max(200, 25 * n_rows)
        dense_param = np.linspace(0.0, polyline_len, dense_count, dtype=np.float32)

        try:
            from scipy.interpolate import CubicSpline

            if np.any(np.diff(s) <= 1e-6):
                raise ValueError("centerline arc-length samples are not strictly increasing")

            spline_x = CubicSpline(s, center_xz[:, 0], bc_type="natural")
            spline_z = CubicSpline(s, center_xz[:, 1], bc_type="natural")
            spline_y = CubicSpline(s, center_y, bc_type="natural")

            dense_x = spline_x(dense_param).astype(np.float32)
            dense_z = spline_z(dense_param).astype(np.float32)
            dense_y = spline_y(dense_param).astype(np.float32)

            base_dense_xyz = np.stack(
                [dense_x, dense_y, dense_z],
                axis=1,
            ).astype(np.float32)

        except Exception as e:
            self.get_logger().warn(
                f"fix_grid CubicSpline failed, using linear interpolation: {e}"
            )

            dense_xz = linear_eval(center_xz, dense_param)
            dense_y = np.interp(dense_param, s, center_y).astype(np.float32)

            base_dense_xyz = np.stack(
                [dense_xz[:, 0], dense_y, dense_xz[:, 1]],
                axis=1,
            ).astype(np.float32)

        if base_dense_xyz.shape[0] < 3:
            return None

        def find_bend_index(path_xyz):
            """Find where the centerline changes direction the most.

            Uses XZ geometry because the bend is in the side profile. Avoids the
            first and last few samples so the detected bend is not an endpoint.
            """
            path_xyz = np.asarray(path_xyz, dtype=np.float32)

            if path_xyz.shape[0] < 5:
                return max(1, path_xyz.shape[0] // 2)

            xz = path_xyz[:, [0, 2]].astype(np.float32)

            v_prev = xz[1:-1] - xz[:-2]
            v_next = xz[2:] - xz[1:-1]

            prev_norm = np.linalg.norm(v_prev, axis=1)
            next_norm = np.linalg.norm(v_next, axis=1)

            valid = (prev_norm > 1e-6) & (next_norm > 1e-6)

            if not np.any(valid):
                return max(1, path_xyz.shape[0] // 2)

            v_prev_unit = v_prev / np.maximum(prev_norm[:, None], 1e-6)
            v_next_unit = v_next / np.maximum(next_norm[:, None], 1e-6)

            cosang = np.sum(v_prev_unit * v_next_unit, axis=1)
            cosang = np.clip(cosang, -1.0, 1.0)

            turn_angle = np.arccos(cosang)

            # Prefer bends near the table/fold, but still mostly use curvature.
            z_mid = path_xyz[1:-1, 2]
            table_weight = np.exp(
                -np.abs(z_mid - float(self.table_z)) / max(2.0 * spacing, 1e-6)
            )

            score = turn_angle * (0.25 + 0.75 * table_weight)

            # Avoid endpoints.
            margin = max(3, int(0.05 * path_xyz.shape[0]))
            score[:margin] = -1.0
            score[-margin:] = -1.0

            if np.max(score) <= 0.0:
                return max(1, path_xyz.shape[0] // 2)

            # score index is for path index 1:-1, so add 1.
            bend_idx = int(np.argmax(score) + 1)
            bend_idx = max(1, min(bend_idx, path_xyz.shape[0] - 2))

            return bend_idx

        bend_idx = find_bend_index(base_dense_xyz)

        def add_bend_detour(input_dense_xyz, fold_len):
            """Insert hidden length at the spline bend/knee.

            The original first endpoint and last endpoint stay the same.
            Extra length is inserted where the path changes direction, usually the
            table-to-upright transition.

            The detour is mostly horizontal/table-parallel so it does not send the
            spline above the grippers.
            """
            input_dense_xyz = np.asarray(input_dense_xyz, dtype=np.float32)

            if fold_len <= 1e-6 or input_dense_xyz.shape[0] < 3:
                return input_dense_xyz.copy()

            idx = max(1, min(int(bend_idx), input_dense_xyz.shape[0] - 2))

            p_bend = input_dense_xyz[idx].copy()
            p_prev = input_dense_xyz[idx - 1].copy()
            p_next = input_dense_xyz[idx + 1].copy()

            # Use the direction entering the bend, projected onto the table.
            # This extends the horizontal/fold part instead of extending the
            # gripper-side tangent upward.
            detour_dir = p_bend - p_prev
            detour_dir = detour_dir.astype(np.float32)
            detour_dir[1] = 0.0
            detour_dir[2] = 0.0

            detour_norm = float(np.linalg.norm(detour_dir))

            if detour_norm < 1e-6:
                # Fallback: use outgoing direction projected onto table.
                detour_dir = p_next - p_bend
                detour_dir = detour_dir.astype(np.float32)
                detour_dir[1] = 0.0
                detour_dir[2] = 0.0
                detour_norm = float(np.linalg.norm(detour_dir))

            if detour_norm < 1e-6:
                # Fallback from row-center geometry near the same relative position.
                row_idx = int(round(idx / max(input_dense_xyz.shape[0] - 1, 1) * (row_centers.shape[0] - 1)))
                row_idx = max(1, min(row_idx, row_centers.shape[0] - 1))

                detour_dir = row_centers[row_idx] - row_centers[row_idx - 1]
                detour_dir = detour_dir.astype(np.float32)
                detour_dir[1] = 0.0
                detour_dir[2] = 0.0
                detour_norm = float(np.linalg.norm(detour_dir))

            if detour_norm < 1e-6:
                # Final fallback. Flip this if your RViz "back" direction is -x.
                detour_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                detour_dir = (detour_dir / detour_norm).astype(np.float32)

            p_detour = p_bend + float(fold_len) * detour_dir
            p_detour[1] = p_bend[1]
            p_detour[2] = float(self.table_z)

            return np.vstack(
                [
                    input_dense_xyz[: idx + 1],
                    p_detour[None, :],
                    input_dense_xyz[idx + 1 :],
                ]
            ).astype(np.float32)

        def prepare_dense_geometry(candidate_dense_xyz):
            """Remove duplicate samples and compute arc-length/tangents."""
            candidate_dense_xyz = np.asarray(candidate_dense_xyz, dtype=np.float32)
            candidate_dense_xyz = candidate_dense_xyz[
                np.all(np.isfinite(candidate_dense_xyz), axis=1)
            ]

            if candidate_dense_xyz.shape[0] < 2:
                return None

            dense_seg_len_local = np.linalg.norm(
                np.diff(candidate_dense_xyz, axis=0),
                axis=1,
            )
            keep_dense_local = np.concatenate([[True], dense_seg_len_local > 1e-7])
            candidate_dense_xyz = candidate_dense_xyz[keep_dense_local]

            if candidate_dense_xyz.shape[0] < 2:
                return None

            dense_seg_len_local = np.linalg.norm(
                np.diff(candidate_dense_xyz, axis=0),
                axis=1,
            )
            dense_arc_local = np.concatenate(
                [[0.0], np.cumsum(dense_seg_len_local)]
            ).astype(np.float32)

            spline_len_local = float(dense_arc_local[-1])

            if spline_len_local < 1e-6:
                return None

            first_ds_local = max(float(dense_arc_local[1] - dense_arc_local[0]), 1e-6)
            last_ds_local = max(float(dense_arc_local[-1] - dense_arc_local[-2]), 1e-6)

            first_tangent_local = (
                candidate_dense_xyz[1] - candidate_dense_xyz[0]
            ) / first_ds_local
            last_tangent_local = (
                candidate_dense_xyz[-1] - candidate_dense_xyz[-2]
            ) / last_ds_local

            first_norm_local = float(np.linalg.norm(first_tangent_local))
            last_norm_local = float(np.linalg.norm(last_tangent_local))

            if first_norm_local < 1e-6 or last_norm_local < 1e-6:
                return None

            first_tangent_local = (
                first_tangent_local / first_norm_local
            ).astype(np.float32)
            last_tangent_local = (
                last_tangent_local / last_norm_local
            ).astype(np.float32)

            return {
                "dense_xyz": candidate_dense_xyz.astype(np.float32),
                "dense_arc": dense_arc_local.astype(np.float32),
                "spline_len": spline_len_local,
                "first_tangent": first_tangent_local,
                "last_tangent": last_tangent_local,
            }

        geom = prepare_dense_geometry(base_dense_xyz)

        if geom is None:
            return None

        dense_xyz = geom["dense_xyz"]
        dense_arc = geom["dense_arc"]
        spline_len = geom["spline_len"]
        first_tangent = geom["first_tangent"]
        last_tangent = geom["last_tangent"]

        def set_current_geometry(candidate_geom):
            nonlocal dense_xyz, dense_arc, spline_len, first_tangent, last_tangent

            dense_xyz = candidate_geom["dense_xyz"]
            dense_arc = candidate_geom["dense_arc"]
            spline_len = candidate_geom["spline_len"]
            first_tangent = candidate_geom["first_tangent"]
            last_tangent = candidate_geom["last_tangent"]

        def eval_center_at_arc(target):
            target = float(target)

            # Temporary negative extrapolation is allowed for bracketing only.
            # The final accepted grid tries to make sampled_arcs[0] >= 0.
            if target < 0.0:
                return (
                    dense_xyz[0] + target * first_tangent
                ).astype(np.float32)

            # Do NOT extend beyond the gripper/end endpoint. Extra length is added
            # at the bend/knee instead.
            if target > spline_len:
                return dense_xyz[-1].copy()

            return np.array(
                [
                    np.interp(target, dense_arc, dense_xyz[:, 0]),
                    np.interp(target, dense_arc, dense_xyz[:, 1]),
                    np.interp(target, dense_arc, dense_xyz[:, 2]),
                ],
                dtype=np.float32,
            )

        def previous_center_at_euclidean_spacing(current_center, current_arc):
            """Find an earlier arc value whose Euclidean distance is spacing."""
            high = float(current_arc)
            low = high - spacing

            def distance_at(arc_value):
                return float(
                    np.linalg.norm(eval_center_at_arc(arc_value) - current_center)
                )

            for _ in range(64):
                if distance_at(low) >= spacing:
                    break
                low -= spacing
            else:
                return None, None

            for _ in range(32):
                mid = 0.5 * (low + high)

                if distance_at(mid) >= spacing:
                    low = mid
                else:
                    high = mid

            sampled_arc = low
            sampled_center = eval_center_at_arc(sampled_arc).astype(np.float32)

            return sampled_center, sampled_arc

        def sample_grid_from_current_geometry():
            """Sample rows backward from the current geometry endpoint."""
            sampled_centers_local = np.zeros((n_rows, 3), dtype=np.float32)
            sampled_arcs_local = np.zeros(n_rows, dtype=np.float32)

            sampled_centers_local[-1] = eval_center_at_arc(spline_len)
            sampled_arcs_local[-1] = float(spline_len)

            for row_idx in range(n_rows - 2, -1, -1):
                center, arc_value = previous_center_at_euclidean_spacing(
                    sampled_centers_local[row_idx + 1],
                    sampled_arcs_local[row_idx + 1],
                )

                if center is None:
                    return None, None

                sampled_centers_local[row_idx] = center
                sampled_arcs_local[row_idx] = arc_value

            return sampled_centers_local, sampled_arcs_local

        sampled_centers, sampled_arcs = sample_grid_from_current_geometry()

        if sampled_centers is None:
            self.get_logger().warn(
                "fix_grid failed initial sampling from fitted spline"
            )
            return None

        chosen_fold_len = 0.0
        chosen_dense_xyz = dense_xyz.copy()

        # If row 0 goes in front of the original first row, inject length at the
        # bend/knee until row 0 is at or behind the original first row.
        if float(sampled_arcs[0]) < -1e-4:
            low_fold = 0.0
            high_fold = max(spacing, -float(sampled_arcs[0]))

            best_centers = sampled_centers
            best_arcs = sampled_arcs
            best_geom = geom
            found_valid = False

            for _ in range(64):
                candidate_dense_xyz = add_bend_detour(
                    base_dense_xyz,
                    high_fold,
                )
                candidate_geom = prepare_dense_geometry(candidate_dense_xyz)

                if candidate_geom is None:
                    high_fold += spacing
                    continue

                set_current_geometry(candidate_geom)
                candidate_centers, candidate_arcs = sample_grid_from_current_geometry()

                if candidate_centers is not None and float(candidate_arcs[0]) >= 0.0:
                    best_centers = candidate_centers
                    best_arcs = candidate_arcs
                    best_geom = candidate_geom
                    found_valid = True
                    break

                high_fold += spacing

            if found_valid:
                for _ in range(40):
                    mid_fold = 0.5 * (low_fold + high_fold)

                    candidate_dense_xyz = add_bend_detour(
                        base_dense_xyz,
                        mid_fold,
                    )
                    candidate_geom = prepare_dense_geometry(candidate_dense_xyz)

                    if candidate_geom is None:
                        low_fold = mid_fold
                        continue

                    set_current_geometry(candidate_geom)
                    candidate_centers, candidate_arcs = sample_grid_from_current_geometry()

                    if candidate_centers is None:
                        low_fold = mid_fold
                        continue

                    if float(candidate_arcs[0]) < 0.0:
                        low_fold = mid_fold
                    else:
                        best_centers = candidate_centers
                        best_arcs = candidate_arcs
                        best_geom = candidate_geom
                        high_fold = mid_fold

                set_current_geometry(best_geom)
                sampled_centers = best_centers
                sampled_arcs = best_arcs
                chosen_fold_len = float(high_fold)
                chosen_dense_xyz = dense_xyz.copy()

            else:
                self.get_logger().warn(
                    "fix_grid could not find valid bend detour; using original sample"
                )

                original_geom = prepare_dense_geometry(base_dense_xyz)
                if original_geom is not None:
                    set_current_geometry(original_geom)
                    chosen_dense_xyz = dense_xyz.copy()

        self.publish_spline_marker(chosen_dense_xyz)

        self.get_logger().info(
            f"fix_grid arcs: front={float(sampled_arcs[0]):.4f}, "
            f"back={float(sampled_arcs[-1]):.4f}, "
            f"spline_len={float(spline_len):.4f}, "
            f"spacing={float(spacing):.4f}, "
            f"bend_idx={int(bend_idx)}, "
            f"bend_detour={float(chosen_fold_len):.4f}"
        )

        # Preserve the original across-cloth direction.
        if self.left_ee_position is not None and self.right_ee_position is not None:
            y_sign = np.sign(self.right_ee_position[1] - self.left_ee_position[1])

            if abs(float(y_sign)) < 1e-6:
                y_sign = 1.0

            width_dir = np.array([0.0, y_sign, 0.0], dtype=np.float32)

        else:
            width_vec = np.mean(grid[:, -1, :] - grid[:, 0, :], axis=0)
            width_norm = float(np.linalg.norm(width_vec))

            if width_norm < 1e-6:
                width_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                width_dir = (width_vec / width_norm).astype(np.float32)

        col_offsets = (
            np.arange(n_cols, dtype=np.float32) - 0.5 * float(n_cols - 1)
        ) * spacing

        fixed_grid = np.zeros((n_rows, n_cols, 3), dtype=np.float32)

        for i in range(n_rows):
            for j in range(n_cols):
                fixed_grid[i, j] = sampled_centers[i] + width_dir * col_offsets[j]

        if points_3d is None:
            if isinstance(grid_result, tuple) and len(grid_result) > 1:
                return fixed_grid, grid_result[1]

            return fixed_grid

        points = np.asarray(points_3d, dtype=np.float32)
        points = points[np.all(np.isfinite(points), axis=1)]

        observed_mask = np.zeros((n_rows, n_cols), dtype=bool)
        radius2 = (0.8 * spacing) ** 2

        if points.shape[0] > 0:
            for i in range(n_rows):
                for j in range(n_cols):
                    d = points - fixed_grid[i, j]
                    observed_mask[i, j] = bool(
                        np.any(np.sum(d * d, axis=1) <= radius2)
                    )

        self.publish_cloth_direction_markers(fixed_grid)

        return fixed_grid, observed_mask

    def publish_cloud(self, points, header, publisher):
        if points is None or len(points) == 0:
            return

        points = np.asarray(points, dtype=np.float32)

        if points.ndim == 3:
            points = points.reshape(-1, 3)

        cloud_points = [
            (float(p[0]), float(p[1]), float(p[2]))
            for p in points
            if np.all(np.isfinite(p))
        ]

        if len(cloud_points) == 0:
            return

        cloud_msg = pc2.create_cloud_xyz32(header, cloud_points)
        publisher.publish(cloud_msg)

    def publish_grid_poses(self, grid, header):
        if grid is None:
            return

        msg = PoseArray()
        msg.header = header

        n = grid.shape[0]

        for i in range(n):
            for j in range(n):
                p = grid[i, j]

                pose = Pose()
                pose.position.x = float(p[0])
                pose.position.y = float(p[1])
                pose.position.z = float(p[2])

                # Store grid index in orientation.
                pose.orientation.x = float(i)
                pose.orientation.y = float(j)
                pose.orientation.z = 0.0
                pose.orientation.w = 1.0

                msg.poses.append(pose)

        self.grid_pose_pub.publish(msg)

    def depth_callback(self, depth_msg):
        if self.stop:
            return

        if self.fx is None:
            self.get_logger().warn("Waiting for camera intrinsics")
            return

        if self.latest_mask is None:
            self.get_logger().warn("No latest SAM cloth mask yet")
            return

        try:
            depth_img = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )

            observed_points = self.observed_points_from_depth(
                depth_img,
                depth_msg.encoding,
            )

            if observed_points is None or len(observed_points) == 0:
                self.get_logger().warn("No valid observed 3D cloth points")
                return

            grid_result = self.build_cloth_grid_from_points(observed_points)

            if grid_result is None:
                self.get_logger().warn("Could not build cloth grid")
                return

            fixed_grid_result = self.fix_grid(grid_result, observed_points)

            if fixed_grid_result is None:
                self.get_logger().warn("Could not fix cloth grid")
                return

            cloth_grid, observed_mask = fixed_grid_result

            out_header = depth_msg.header
            out_header.frame_id = self.output_frame_id

            self.publish_cloud(
                observed_points,
                out_header,
                self.observed_points_pub,
            )

            self.publish_cloud(
                cloth_grid,
                out_header,
                self.grid_cloud_pub,
            )

            self.publish_grid_poses(
                cloth_grid,
                out_header,
            )

            self.get_logger().info(
                f"Published cloth grid: {self.cloth_grid_n}x{self.cloth_grid_n}, "
                f"spacing={self.cloth_spacing_m:.3f}m, "
                f"observed_cells={int(np.sum(observed_mask))}, "
                f"observed_points={len(observed_points)}"
            )

        except Exception as e:
            self.get_logger().error(f"Depth callback failed: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = SingleCameraSAMClothGridNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()