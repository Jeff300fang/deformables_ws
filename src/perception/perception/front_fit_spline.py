#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import rclpy

from rclpy.node import Node
from cv_bridge import CvBridge
from scipy.interpolate import splprep, splev

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Float32MultiArray
import sensor_msgs_py.point_cloud2 as pc2
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Bool

class TAPNextDepthSplineNode(Node):
    def __init__(self):
        super().__init__("front_tapnext_depth_spline_node")

        self.declare_parameter("keypoints_topic", "/front_camera/tapnext/keypoints")
        self.declare_parameter("depth_topic", "/front_camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/front_camera/color/camera_info")

        self.declare_parameter("points_3d_topic", "/front/tapnext/rope_points_3d")
        self.declare_parameter("spline_3d_topic", "/front/tapnext/rope_spline_3d")

        self.declare_parameter("depth_search_radius", 4)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 5.0)

        self.declare_parameter("spline_points", 300)
        self.declare_parameter("temporal_alpha", 0.65)
        self.declare_parameter("max_point_jump_m", 0.05)

        self.declare_parameter("body_translation_x", 0.0)
        self.declare_parameter("body_translation_y", 0.0)
        self.declare_parameter("body_translation_z", 0.0)

        self.body_translation_x = float(
            self.get_parameter("body_translation_x").value
        )

        self.body_translation_y = float(
            self.get_parameter("body_translation_y").value
        )

        self.body_translation_z = float(
            self.get_parameter("body_translation_z").value
        )
        self.add_on_set_parameters_callback(self.parameter_callback)

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
        self.prev_spline_3d = None

        self.info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            1,
        )

        self.keypoints_sub = self.create_subscription(
            Float32MultiArray,
            self.keypoints_topic,
            self.keypoints_callback,
            1,
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            1,
        )

        self.points_pub = self.create_publisher(
            PointCloud2,
            self.points_3d_topic,
            1,
        )

        self.spline_pub = self.create_publisher(
            PointCloud2,
            self.spline_3d_topic,
            1,
        )

        self.rope_pose_pub = self.create_publisher(
            PoseArray,
            "/front/rope_poses",
            1,
        )

        self.stop_front_sub = self.create_subscription(
            Bool,
            '/stop_front',
            self.stop_callback,
            1
        )
        self.stop = False

        self.get_logger().info(f"Subscribed TAPNext keypoints: {self.keypoints_topic}")
        self.get_logger().info(f"Subscribed depth: {self.depth_topic}")
        self.get_logger().info(f"Subscribed camera info: {self.camera_info_topic}")
        self.get_logger().info(f"Publishing raw 3D points: {self.points_3d_topic}")
        self.get_logger().info(f"Publishing spline 3D points: {self.spline_3d_topic}")

    def stop_callback(self, msg):
        self.stop = msg.data

    def sample_centered_by_euclidean_distance(self, points, spacing_m=0.1, num_segments=9):
        points = np.asarray(points, dtype=np.float32)

        if points.shape[0] == 0:
            return points

        if points.shape[0] == 1:
            return points.copy()

        diffs = np.diff(points, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        total_len = cumlen[-1]
        if total_len < 1e-8:
            return points[:1].copy()

        num_samples = num_segments + 1
        sample_length = (num_samples - 1) * spacing_m

        start_s = 0.5 * total_len - 0.5 * sample_length
        end_s = start_s + sample_length

        # Clamp if requested sampled rope is longer than available spline
        if start_s < 0.0:
            start_s = 0.0
            end_s = min(sample_length, total_len)

        if end_s > total_len:
            end_s = total_len
            start_s = max(0.0, total_len - sample_length)

        target_s = np.linspace(start_s, end_s, num_samples)

        sampled = []
        for s in target_s:
            idx = np.searchsorted(cumlen, s, side="right") - 1
            idx = np.clip(idx, 0, len(points) - 2)

            seg_len = seg_lengths[idx]
            if seg_len < 1e-8:
                sampled.append(points[idx].copy())
                continue

            ratio = (s - cumlen[idx]) / seg_len
            p = points[idx] + ratio * (points[idx + 1] - points[idx])
            sampled.append(p.astype(np.float32))

        return np.asarray(sampled, dtype=np.float32)

    def parameter_callback(self, params):
        try:
            for param in params:
                if param.name == "depth_search_radius":
                    self.depth_search_radius = int(param.value)

                elif param.name == "min_depth_m":
                    self.min_depth_m = float(param.value)

                elif param.name == "max_depth_m":
                    self.max_depth_m = float(param.value)

                elif param.name == "spline_points":
                    self.spline_points = int(param.value)

                elif param.name == "temporal_alpha":
                    self.temporal_alpha = float(param.value)

                elif param.name == "max_point_jump_m":
                    self.max_point_jump_m = float(param.value)

                elif param.name == "body_translation_x":
                    self.body_translation_x = float(param.value)

                elif param.name == "body_translation_y":
                    self.body_translation_y = float(param.value)

                elif param.name == "body_translation_z":
                    self.body_translation_z = float(param.value)

            return SetParametersResult(successful=True)

        except Exception as e:
            return SetParametersResult(successful=False, reason=str(e))

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

        # self.get_logger().info(
        #     f"Received TAPNext keypoints: {len(self.latest_xy)}"
        # )

    def depth_to_meters(self, depth_value, encoding):
        if encoding in ("16UC1", "mono16"):
            return float(depth_value) * 0.001

        if encoding == "32FC1":
            return float(depth_value)

        return float(depth_value)

    def temporal_smooth_spline(self, spline_3d, alpha=0.65):
        spline_3d = np.asarray(spline_3d, dtype=np.float32)

        if self.prev_spline_3d is None:
            self.prev_spline_3d = spline_3d
            return spline_3d

        if self.prev_spline_3d.shape != spline_3d.shape:
            self.prev_spline_3d = spline_3d
            return spline_3d

        smoothed = (
            alpha * self.prev_spline_3d
            + (1.0 - alpha) * spline_3d
        )

        self.prev_spline_3d = smoothed

        return smoothed

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

    def filter_median_depth_outliers(
        self,
        points_3d,
        max_depth_offset_m=0.15,
    ):
        points_3d = np.asarray(points_3d, dtype=np.float32)

        if points_3d.shape[0] == 0:
            return points_3d

        z = points_3d[:, 2]

        median_z = np.median(z)

        keep = np.abs(z - median_z) <= max_depth_offset_m

        filtered = points_3d[keep]

        # self.get_logger().info(
        #     f"Median depth filter: "
        #     f"median_z={median_z:.3f}m, "
        #     f"kept {len(filtered)}/{len(points_3d)} points"
        # )

        return filtered

    def sample_by_exact_euclidean_distance(self, points, spacing_m=0.1, num_segments=15):
        points = np.asarray(points, dtype=np.float32)

        if points.shape[0] == 0:
            return points

        sampled = [points[0].copy()]
        last = points[0].astype(np.float32)

        i = 1
        while i < len(points) and len(sampled) < num_segments + 1:
            p = points[i].astype(np.float32)

            v = p - last
            dist = np.linalg.norm(v)

            if dist < 1e-8:
                i += 1
                continue

            if dist >= spacing_m:
                new_p = last + (spacing_m / dist) * v
                sampled.append(new_p.astype(np.float32))
                last = new_p.astype(np.float32)

                # Do not increment i.
                # There may still be room before points[i].
            else:
                i += 1

        return np.asarray(sampled, dtype=np.float32)

    def fit_spline_3d(self, points_3d):
        points_3d = np.asarray(points_3d, dtype=np.float32)

        if points_3d.shape[0] < 4:
            return points_3d

        # Remove duplicate / nearly duplicate points
        diffs = np.linalg.norm(np.diff(points_3d, axis=0), axis=1)
        keep = np.concatenate([[True], diffs > 1e-5])
        points_3d = points_3d[keep]

        if points_3d.shape[0] < 4:
            return points_3d

        k = min(3, points_3d.shape[0] - 1)

        try:
            tck, _ = splprep(
                [
                    points_3d[:, 0],
                    points_3d[:, 1],
                    points_3d[:, 2],
                ],
                k=k,
                s=0.1,
            )

            u_new = np.linspace(0.0, 1.0, self.spline_points)

            x_new, y_new, z_new = splev(u_new, tck)

            spline = np.stack(
                [x_new, y_new, z_new],
                axis=1,
            ).astype(np.float32)

            return spline

        except Exception as e:
            self.get_logger().warn(
                f"Cubic spline fit failed, falling back to raw points: {e}"
            )
            return points_3d

    def optical_to_body_frame(self, points_3d):
        points_3d = np.asarray(points_3d, dtype=np.float32)

        if points_3d.shape[0] == 0:
            return points_3d

        x_opt = points_3d[:, 0]
        y_opt = points_3d[:, 1]
        z_opt = points_3d[:, 2]

        points_body = np.zeros_like(points_3d)

        # optical -> body
        points_body[:, 0] = z_opt
        points_body[:, 1] = -x_opt
        points_body[:, 2] = -y_opt

        # additional 180 deg rotation about z-axis
        points_body[:, 0] *= -1
        points_body[:, 1] *= -1

        # translation
        points_body[:, 0] += self.body_translation_x
        points_body[:, 1] += self.body_translation_y
        points_body[:, 2] += self.body_translation_z

        return points_body

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

    def smooth_spline_points(self, spline_3d, window_size=15):
        spline_3d = np.asarray(spline_3d, dtype=np.float32)

        if spline_3d.shape[0] < window_size:
            return spline_3d

        if window_size % 2 == 0:
            window_size += 1

        kernel = np.ones(window_size, dtype=np.float32)
        kernel /= np.sum(kernel)

        pad = window_size // 2
        padded = np.pad(
            spline_3d,
            ((pad, pad), (0, 0)),
            mode="edge",
        )

        smoothed = np.zeros_like(spline_3d)

        for dim in range(3):
            smoothed[:, dim] = np.convolve(
                padded[:, dim],
                kernel,
                mode="valid",
            )

        smoothed[0] = spline_3d[0]
        smoothed[-1] = spline_3d[-1]

        return smoothed

    def depth_callback(self, depth_msg):
        if self.stop:
            return
        
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

            # self.get_logger().info(
            #     f"Received {len(xy)} TAPNext points, valid depth points={len(points_3d)}"
            # )

            if len(points_3d) < 2:
                self.get_logger().warn("Not enough valid depth keypoints")
                return

            points_3d = np.asarray(points_3d, dtype=np.float32)

            before_filter = len(points_3d)

            points_3d = self.filter_median_depth_outliers(
                points_3d,
                max_depth_offset_m=0.15,
            )

            after_filter = len(points_3d)

            # self.get_logger().info(
            #     f"Consecutive-gap filter kept "
            #     f"{after_filter}/{before_filter} points"
            # )

            if len(points_3d) < 2:
                self.get_logger().warn(
                    "Not enough points after filtering"
                )
                return

            points_3d = self.filter_and_smooth_points(points_3d)


            spline_3d = self.fit_spline_3d(points_3d)

            spline_3d = self.smooth_spline_points(
                spline_3d,
                window_size=15,
            )

            spline_3d = self.temporal_smooth_spline(
                spline_3d,
                alpha=self.temporal_alpha,
            )

            # self.get_logger().info(
            #     f"Publishing {len(points_3d)} raw 3D points "
            #     f"and {len(spline_3d)} spline points"
            # )

            points_3d = self.optical_to_body_frame(points_3d)
            spline_3d = self.optical_to_body_frame(spline_3d)


            rope_pose_points = self.sample_centered_by_euclidean_distance(
                spline_3d,
                spacing_m=0.1,
                num_segments=9,
            )

            out_header = depth_msg.header
            out_header.frame_id = "workstation"

            self.publish_cloud(points_3d, out_header, self.points_pub)
            self.publish_cloud(spline_3d, out_header, self.spline_pub)
            self.publish_rope_poses(
                rope_pose_points,
                out_header,
            )

        except Exception as e:
            self.get_logger().error(f"Depth spline callback failed: {e}")

    def publish_rope_poses(self, points, header):
        msg = PoseArray()
        msg.header = header

        points = np.asarray(points, dtype=np.float32)

        for p in points:
            pose = Pose()

            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = float(p[2])

            pose.orientation.w = 1.0

            msg.poses.append(pose)

        self.rope_pose_pub.publish(msg)

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