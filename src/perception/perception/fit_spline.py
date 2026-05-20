#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import rclpy

from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Float32MultiArray
import sensor_msgs_py.point_cloud2 as pc2


class TAPNextDepthSplineNode(Node):
    def __init__(self):
        super().__init__("tapnext_depth_spline_node")

        self.declare_parameter("keypoints_topic", "/front_camera/tapnext/keypoints")
        self.declare_parameter("depth_topic", "/front_camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/front_camera/color/camera_info")

        self.declare_parameter("points_3d_topic", "/tapnext/rope_points_3d")
        self.declare_parameter("spline_3d_topic", "/tapnext/rope_spline_3d")

        self.declare_parameter("depth_search_radius", 4)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 5.0)

        self.declare_parameter("spline_points", 100)
        self.declare_parameter("temporal_alpha", 0.65)
        self.declare_parameter("max_point_jump_m", 0.15)

        self.keypoints_topic = str(self.get_parameter("keypoints_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)

        self.points_3d_topic = str(self.get_parameter("points_3d_topic").value)
        self.spline_3d_topic = str(self.get_parameter("spline_3d_topic").value)

        self.depth_search_radius = int(self.get_parameter("depth_search_radius").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        self.spline_points = int(self.get_parameter("spline_points").value)
        self.temporal_alpha = float(self.get_parameter("temporal_alpha").value)
        self.max_point_jump_m = float(self.get_parameter("max_point_jump_m").value)

        self.bridge = CvBridge()

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.latest_xy = None
        self.prev_points_3d = None

        self.info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.keypoints_sub = self.create_subscription(
            Float32MultiArray,
            self.keypoints_topic,
            self.keypoints_callback,
            10,
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            10,
        )

        self.points_pub = self.create_publisher(
            PointCloud2,
            self.points_3d_topic,
            10,
        )

        self.spline_pub = self.create_publisher(
            PointCloud2,
            self.spline_3d_topic,
            10,
        )

        self.get_logger().info(f"Subscribed TAPNext keypoints: {self.keypoints_topic}")
        self.get_logger().info(f"Subscribed depth: {self.depth_topic}")
        self.get_logger().info(f"Subscribed camera info: {self.camera_info_topic}")
        self.get_logger().info(f"Publishing raw 3D points: {self.points_3d_topic}")
        self.get_logger().info(f"Publishing spline 3D points: {self.spline_3d_topic}")

    def camera_info_callback(self, msg):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def keypoints_callback(self, msg):
        data = np.asarray(msg.data, dtype=np.float32)

        if data.size == 0:
            self.latest_xy = None
            return

        if data.size % 3 == 0:
            arr = data.reshape(-1, 3)

            xy = arr[:, :2]
            visible = arr[:, 2] > 0.5

            self.latest_xy = xy[visible]

        elif data.size % 2 == 0:
            self.latest_xy = data.reshape(-1, 2)

        else:
            self.latest_xy = None
            self.get_logger().warn(
                f"Invalid TAPNext Float32MultiArray size: {data.size}"
            )
            return

        self.get_logger().info(
            f"Received TAPNext keypoints: {len(self.latest_xy)}"
        )

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

    def filter_and_smooth_points(self, points_3d):
        if points_3d is None or len(points_3d) == 0:
            return None

        points_3d = np.asarray(points_3d, dtype=np.float32)

        if self.prev_points_3d is None:
            self.prev_points_3d = points_3d
            return points_3d

        if self.prev_points_3d.shape != points_3d.shape:
            self.prev_points_3d = points_3d
            return points_3d

        deltas = np.linalg.norm(points_3d - self.prev_points_3d, axis=1)
        valid = deltas < self.max_point_jump_m

        smoothed = points_3d.copy()

        smoothed[valid] = (
            self.temporal_alpha * self.prev_points_3d[valid]
            + (1.0 - self.temporal_alpha) * points_3d[valid]
        )

        self.prev_points_3d = smoothed

        return smoothed

    def fit_spline_3d(self, points_3d):
        points_3d = np.asarray(points_3d, dtype=np.float32)

        if points_3d.shape[0] < 2:
            return points_3d

        diffs = np.diff(points_3d, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)

        keep = np.concatenate([[True], seg_lengths > 1e-6])
        points_3d = points_3d[keep]

        if points_3d.shape[0] < 2:
            return points_3d

        diffs = np.diff(points_3d, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)

        cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_length = cumulative[-1]

        if total_length < 1e-6:
            return points_3d

        t = cumulative / total_length
        target_t = np.linspace(0.0, 1.0, self.spline_points)

        spline = np.zeros((self.spline_points, 3), dtype=np.float32)
        spline[:, 0] = np.interp(target_t, t, points_3d[:, 0])
        spline[:, 1] = np.interp(target_t, t, points_3d[:, 1])
        spline[:, 2] = np.interp(target_t, t, points_3d[:, 2])

        if spline.shape[0] >= 5:
            kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
            kernel /= np.sum(kernel)

            padded = np.pad(spline, ((2, 2), (0, 0)), mode="edge")

            smoothed = np.zeros_like(spline)
            for i in range(spline.shape[0]):
                window = padded[i:i + 5]
                smoothed[i] = np.sum(window * kernel[:, None], axis=0)

            smoothed[0] = spline[0]
            smoothed[-1] = spline[-1]
            spline = smoothed

        return spline

    def publish_cloud(self, points, header, publisher):
        if points is None or len(points) == 0:
            return

        points = np.asarray(points, dtype=np.float32)

        cloud_points = [
            (float(p[0]), float(p[1]), float(p[2]))
            for p in points
            if np.all(np.isfinite(p))
        ]

        if len(cloud_points) == 0:
            return

        cloud_msg = pc2.create_cloud_xyz32(header, cloud_points)
        publisher.publish(cloud_msg)

    def depth_callback(self, depth_msg):
        if self.fx is None:
            self.get_logger().warn("Waiting for camera intrinsics")
            return

        if self.latest_xy is None or len(self.latest_xy) == 0:
            self.get_logger().warn("No latest TAPNext keypoints yet")
            return

        try:
            depth_img = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )

            xy = self.latest_xy

            points_3d = []

            for x, y in xy:
                z = self.get_valid_depth(
                    depth_img,
                    x,
                    y,
                    depth_msg.encoding,
                )

                if z is None:
                    continue

                p3d = self.pixel_to_3d(x, y, z)
                points_3d.append(p3d)

            self.get_logger().info(
                f"Received {len(xy)} TAPNext points, valid depth points={len(points_3d)}"
            )

            if len(points_3d) < 2:
                self.get_logger().warn("Not enough valid depth keypoints")
                return

            points_3d = np.asarray(points_3d, dtype=np.float32)
            points_3d = self.filter_and_smooth_points(points_3d)

            spline_3d = self.fit_spline_3d(points_3d)

            self.get_logger().info(
                f"Publishing {len(points_3d)} raw 3D points "
                f"and {len(spline_3d)} spline points"
            )

            out_header = depth_msg.header

            self.publish_cloud(points_3d, out_header, self.points_pub)
            self.publish_cloud(spline_3d, out_header, self.spline_pub)

        except Exception as e:
            self.get_logger().error(f"Depth spline callback failed: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = TAPNextDepthSplineNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()