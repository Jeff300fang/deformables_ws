#!/usr/bin/env python3

import sys
from pathlib import Path
from collections import deque

import cv2 as cv
import numpy as np
import torch
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge


for workspace_root in [Path.cwd(), *Path(__file__).resolve().parents]:
    for src_dir in (workspace_root, workspace_root / "src"):
        local_package_path = src_dir / "tapnet"
        if local_package_path.exists() and str(local_package_path) not in sys.path:
            sys.path.insert(0, str(local_package_path))

from tapnet.tapnext.tapnext_torch import TAPNext


CKPT_SIZE = (256, 256)


class TAPNextFromSAMNode(Node):
    def __init__(self):
        super().__init__("tapnext_from_sam_node")

        self.declare_parameter("image_topic", "/front_camera/color/image_raw")
        self.declare_parameter("sam_keypoints_topic", "/front_camera/sam_rope/keypoints")
        self.declare_parameter("tracked_keypoints_topic", "/front_camera/tapnext/keypoints")
        self.declare_parameter(
            "annotated_topic",
            "/front_camera/tapnext/annotated_image/compressed",
        )
        self.declare_parameter(
            "tapnn_checkpoint_path",
            "/home/jeff/trustworthroboticsgroup/CoRL2026/deformables_ws/src/tapnet/checkpoints/tapnextpp_ckpt.pt",
        )

        self.declare_parameter("max_sam_reinit_rate_hz", 1.0)
        self.declare_parameter("image_buffer_size", 90)
        self.declare_parameter("publish_annotated", True)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.sam_keypoints_topic = str(self.get_parameter("sam_keypoints_topic").value)
        self.tracked_keypoints_topic = str(
            self.get_parameter("tracked_keypoints_topic").value
        )
        self.annotated_topic = str(self.get_parameter("annotated_topic").value)
        self.ckpt_path = str(self.get_parameter("tapnn_checkpoint_path").value)

        self.max_sam_reinit_rate_hz = float(
            self.get_parameter("max_sam_reinit_rate_hz").value
        )
        self.image_buffer_size = int(self.get_parameter("image_buffer_size").value)
        self.publish_annotated_enabled = bool(
            self.get_parameter("publish_annotated").value
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.get_logger().info(f"Using device: {self.device}")

        self.model = TAPNext(image_size=CKPT_SIZE)

        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        self.model.load_state_dict(
            {k.replace("tapnext.", ""): v for k, v in ckpt["state_dict"].items()}
        )
        self.model = self.model.to(self.device).eval()

        self.bridge = CvBridge()

        self.initialized = False
        self.tracking_state = None
        self.last_reinit_time = self.get_clock().now()

        self.image_buffer = {}
        self.image_order = deque()

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.keypoints_sub = self.create_subscription(
            PoseArray,
            self.sam_keypoints_topic,
            self.keypoints_callback,
            10,
        )

        self.keypoints_pub = self.create_publisher(
            Float32MultiArray,
            self.tracked_keypoints_topic,
            10,
        )

        self.annotated_pub = self.create_publisher(
            CompressedImage,
            self.annotated_topic,
            10,
        )

        self.get_logger().info(f"Subscribed image: {self.image_topic}")
        self.get_logger().info(f"Subscribed SAM keypoints: {self.sam_keypoints_topic}")
        self.get_logger().info(
            f"Publishing TAPNext keypoints: {self.tracked_keypoints_topic}"
        )

    def stamp_to_ns(self, stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def should_accept_sam_reinit(self):
        if self.max_sam_reinit_rate_hz <= 0.0:
            return True

        now = self.get_clock().now()
        dt = (now - self.last_reinit_time).nanoseconds * 1e-9

        if dt < 1.0 / self.max_sam_reinit_rate_hz:
            return False

        self.last_reinit_time = now
        return True

    def store_image(self, msg, frame_rgb):
        stamp_ns = self.stamp_to_ns(msg.header.stamp)

        self.image_buffer[stamp_ns] = frame_rgb
        self.image_order.append(stamp_ns)

        while len(self.image_order) > self.image_buffer_size:
            old_stamp = self.image_order.popleft()
            self.image_buffer.pop(old_stamp, None)

    def keypoints_callback(self, msg):
        if len(msg.poses) == 0:
            self.get_logger().warn("Received empty SAM PoseArray")
            return

        if not self.should_accept_sam_reinit():
            return

        stamp_ns = self.stamp_to_ns(msg.header.stamp)

        if stamp_ns not in self.image_buffer:
            self.get_logger().warn(
                "Received SAM keypoints, but matching image timestamp is not in buffer"
            )
            return

        points = []

        for pose in msg.poses:
            idx = int(round(pose.position.z))
            x = float(pose.position.x)
            y = float(pose.position.y)
            points.append((idx, x, y))

        points.sort(key=lambda p: p[0])

        query_points = np.zeros((len(points), 3), dtype=np.float32)

        for out_i, (_, x, y) in enumerate(points):
            query_points[out_i, 0] = 0.0
            query_points[out_i, 1] = y
            query_points[out_i, 2] = x

        frame_rgb = self.image_buffer[stamp_ns]

        ok = self.initialize_tracker_from_frame(frame_rgb, query_points)

        if ok:
            self.get_logger().info(
                f"Initialized TAPNext using matching SAM frame. points={len(points)}"
            )

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

    def query_points_to_crop_model(self, query_points_orig, crop_info):
        startx, starty, crop_size = crop_info
        model_h, model_w = CKPT_SIZE

        qp = query_points_orig.copy()

        qp[:, 1] -= float(starty)
        qp[:, 2] -= float(startx)

        valid = (
            (qp[:, 1] >= 0)
            & (qp[:, 1] < crop_size)
            & (qp[:, 2] >= 0)
            & (qp[:, 2] < crop_size)
        )

        qp = qp[valid]

        if qp.shape[0] == 0:
            self.get_logger().warn("No SAM keypoints inside TAPNext crop")
            return None

        qp[:, 1] *= model_h / float(crop_size)
        qp[:, 2] *= model_w / float(crop_size)

        return qp[None]

    def initialize_tracker_from_frame(self, frame_rgb, query_points_orig):
        _, _, video, crop_info = self.preprocess_frame(frame_rgb)

        query_points_np = self.query_points_to_crop_model(
            query_points_orig,
            crop_info,
        )

        if query_points_np is None:
            return False

        query_points = torch.from_numpy(query_points_np).to(self.device)

        with torch.no_grad():
            _, _, _, self.tracking_state = self.model(
                video=video,
                query_points=query_points,
            )

        self.initialized = True
        return True

    def step_tracker(self, video):
        with torch.no_grad():
            pred_tracks, _, visible_logits, self.tracking_state = self.model(
                video=video,
                state=self.tracking_state,
            )

        tracks_yx = pred_tracks.cpu().numpy()[0, 0]
        visible = (visible_logits > 0).cpu().numpy()[0, 0, :, 0]

        return tracks_yx, visible

    def model_tracks_to_original_xy(self, tracks_yx, crop_info):
        startx, starty, crop_size = crop_info
        model_h, model_w = CKPT_SIZE

        tracks_xy = tracks_yx[:, ::-1].copy()

        tracks_xy[:, 0] *= crop_size / float(model_w)
        tracks_xy[:, 1] *= crop_size / float(model_h)

        tracks_xy[:, 0] += float(startx)
        tracks_xy[:, 1] += float(starty)

        return tracks_xy

    def publish_keypoints(self, tracks_xy, visible):
        packed = np.concatenate(
            [
                tracks_xy.astype(np.float32),
                visible.reshape(-1, 1).astype(np.float32),
            ],
            axis=1,
        )

        msg = Float32MultiArray()
        msg.data = packed.reshape(-1).tolist()
        self.keypoints_pub.publish(msg)

    def draw_tracks(self, frame_rgb, tracks_xy, visible):
        vis = frame_rgb.copy()

        for i, (xy, is_visible) in enumerate(zip(tracks_xy, visible)):
            if not is_visible:
                continue

            x = int(round(float(xy[0])))
            y = int(round(float(xy[1])))

            cv.circle(vis, (x, y), 5, (255, 0, 255), -1)
            cv.putText(
                vis,
                str(i),
                (x + 6, y - 6),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        for i in range(len(tracks_xy) - 1):
            if not visible[i] or not visible[i + 1]:
                continue

            p0 = tuple(np.round(tracks_xy[i]).astype(int))
            p1 = tuple(np.round(tracks_xy[i + 1]).astype(int))
            cv.line(vis, p0, p1, (255, 255, 0), 2)

        return vis

    def publish_annotated(self, rgb_img, header):
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

        self.annotated_pub.publish(msg)

    def image_callback(self, msg):
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)

            self.store_image(msg, frame_rgb)

            if not self.initialized:
                return

            _, _, video, crop_info = self.preprocess_frame(frame_rgb)

            tracks_yx, visible = self.step_tracker(video)

            tracks_xy = self.model_tracks_to_original_xy(tracks_yx, crop_info)

            self.publish_keypoints(tracks_xy, visible)

            if self.publish_annotated_enabled:
                annotated = self.draw_tracks(frame_rgb, tracks_xy, visible)
                self.publish_annotated(annotated, msg.header)

        except Exception as e:
            self.get_logger().error(f"TAPNext callback failed: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = TAPNextFromSAMNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()