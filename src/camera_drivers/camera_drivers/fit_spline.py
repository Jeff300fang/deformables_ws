#!/usr/bin/env python3

import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

from scipy.interpolate import splprep, splev


class PointCloudSplineFitter(Node):
    def __init__(self):
        super().__init__("pointcloud_spline_fitter")

        self.input_topic = "/tapnext/points_3d"
        self.output_topic = "/tapnext/spline_points"

        self.num_spline_points = 300
        self.smoothing = 0.001

        self.sub = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.pointcloud_callback,
            10,
        )

        self.pub = self.create_publisher(
            PointCloud2,
            self.output_topic,
            10,
        )

        self.get_logger().info(f"Subscribed to: {self.input_topic}")
        self.get_logger().info(f"Publishing spline to: {self.output_topic}")
        self.max_distance = 3.0

    def pointcloud_to_numpy(self, msg: PointCloud2):
        pts = []

        for p in pc2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        ):
            pts.append([float(p[0]), float(p[1]), float(p[2])])

        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float32)

        return np.asarray(pts, dtype=np.float32)

    def sort_points_for_rope(self, points):
        """
        Sort points into a rough rope order.

        Assumption:
          The rope is mostly extended along its largest spatial axis.

        This is simple and works well if the points are already roughly ordered
        or the rope does not strongly loop back on itself.
        """
        if points.shape[0] <= 2:
            return points

        center = np.mean(points, axis=0)
        centered = points - center

        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        main_axis = vh[0]

        scalar = centered @ main_axis
        order = np.argsort(scalar)

        return points[order]


    def fit_spline(self, points):
        """
        Fit a smooth 3D spline that respects the given keypoint order.

        The order of points is assumed to already be the rope order.
        Smoothness is controlled by self.smoothing.
        Larger self.smoothing => smoother curve, less exact interpolation.
        """
        points = np.asarray(points, dtype=np.float32)
        n = points.shape[0]

        if n < 2:
            return points

        # Remove consecutive duplicate / near-duplicate points while preserving order.
        filtered = [points[0]]
        for p in points[1:]:
            if np.linalg.norm(p - filtered[-1]) > 1e-5:
                filtered.append(p)

        points = np.asarray(filtered, dtype=np.float32)
        n = points.shape[0]

        if n < 2:
            return points

        if n == 2:
            t = np.linspace(0.0, 1.0, self.num_spline_points)
            spline = (1.0 - t[:, None]) * points[0] + t[:, None] * points[1]
            return spline.astype(np.float32)

        # Chord-length parameterization.
        # This makes the spline follow the ordered keypoints spatially instead of
        # treating every keypoint as equally spaced.
        seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        u = np.zeros(n, dtype=np.float32)
        u[1:] = np.cumsum(seg_lengths)

        total_len = u[-1]
        if total_len < 1e-6:
            return points

        u = u / total_len

        # Cubic if enough points, otherwise lower-order.
        k = min(3, n - 1)

        try:
            tck, _ = splprep(
                [points[:, 0], points[:, 1], points[:, 2]],
                u=u,
                s=self.smoothing,
                k=k,
            )

            u_new = np.linspace(0.0, 1.0, self.num_spline_points)
            x_new, y_new, z_new = splev(u_new, tck)

            spline_points = np.stack([x_new, y_new, z_new], axis=1)
            return spline_points.astype(np.float32)

        except Exception as e:
            self.get_logger().warn(f"Spline fit failed, using raw points: {e}")
            return points.astype(np.float32)

    def filter_points_by_distance(self, points):
        """
        Keep only points within max_distance from the sensor origin.
        """
        if points.shape[0] == 0:
            return points

        dists = np.linalg.norm(points, axis=1)
        mask = dists <= self.max_distance
        return points[mask]

    def pointcloud_callback(self, msg: PointCloud2):
        points = self.pointcloud_to_numpy(msg)

        if points.shape[0] == 0:
            self.get_logger().warn("Received empty point cloud")
            return

        points = self.filter_points_by_distance(points)

        if points.shape[0] == 0:
            self.get_logger().warn(f"All points filtered out (>{self.max_distance:.2f}m)")
            return

        # IMPORTANT:
        # Do not PCA-sort. The points are already ordered by keypoint index.
        ordered_points = points

        # ordered_points = self.filter_bad_neighbor_jumps(ordered_points)

        if ordered_points.shape[0] < 2:
            self.get_logger().warn("Too few ordered points after filtering")
            return

        spline_points = self.fit_spline(ordered_points)

        out_msg = pc2.create_cloud_xyz32(
            msg.header,
            spline_points.tolist(),
        )

        self.pub.publish(out_msg)

        self.get_logger().info(
            f"Filtered ordered points: {ordered_points.shape[0]}, "
            f"spline points: {spline_points.shape[0]}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = PointCloudSplineFitter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()