#!/usr/bin/env python3

import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
import sensor_msgs_py.point_cloud2 as pc2

from scipy.interpolate import splprep, splev
from collections import deque


class TwoCloudRopeSplineFitter(Node):
    def __init__(self):
        super().__init__("two_cloud_rope_spline_fitter")

        self.cam1_pointcloud_topic = "/tapnext/cam1/points_3d"
        self.cam2_pointcloud_topic = "/tapnext/cam2/points_3d"
        self.output_topic = "/tapnext/spline_points"

        self.num_output_points = 300
        self.smoothing = 1e-1

        # If True: assumes cam1[i] corresponds to cam2[i]
        # If False: concatenates both point clouds
        self.is_ordered = False

        self.cam1_sub = Subscriber(
            self,
            PointCloud2,
            self.cam1_pointcloud_topic,
        )

        self.cam2_sub = Subscriber(
            self,
            PointCloud2,
            self.cam2_pointcloud_topic,
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.cam1_sub, self.cam2_sub],
            queue_size=10,
            slop=0.08,
        )
        self.sync.registerCallback(self.synced_callback)

        self.pub = self.create_publisher(
            PointCloud2,
            self.output_topic,
            10,
        )

        self.target_rope_length = 0.75  # meters, set this to your rope length
        self.enforce_length = True
        self.length_tolerance = 0.03    # meters

        self.history_len = 4
        self.point_history = deque(maxlen=self.history_len)

        self.temporal_alpha = 0.55
        self.prev_filtered_curve = None
        self.max_interframe_motion = 0.05  # meters

        self.raw_history_len = 4
        self.raw_point_history = deque(maxlen=self.raw_history_len)
        self.prev_filtered_raw_points = None

        self.raw_temporal_alpha = 0.55
        self.raw_max_interframe_motion = 0.05

        self.get_logger().info(f"Subscribed to: {self.cam1_pointcloud_topic}")
        self.get_logger().info(f"Subscribed to: {self.cam2_pointcloud_topic}")
        self.get_logger().info(f"Publishing spline to: {self.output_topic}")

    def temporal_filter_curve(self, curve):
        curve = np.asarray(curve, dtype=np.float32)

        if curve.shape[0] == 0:
            return curve

        if self.prev_filtered_curve is None:
            self.prev_filtered_curve = curve.copy()
            self.point_history.append(curve.copy())
            return curve

        if self.prev_filtered_curve.shape != curve.shape:
            self.prev_filtered_curve = curve.copy()
            self.point_history.clear()
            self.point_history.append(curve.copy())
            return curve

        prev = self.prev_filtered_curve

        # Prevent start/end flip
        forward_cost = np.mean(np.linalg.norm(curve - prev, axis=1))
        reverse_cost = np.mean(np.linalg.norm(curve[::-1] - prev, axis=1))

        if reverse_cost < forward_cost:
            curve = curve[::-1]

        disp = np.linalg.norm(curve - prev, axis=1)
        stable_curve = curve.copy()

        jump_mask = disp > self.max_interframe_motion
        stable_curve[jump_mask] = prev[jump_mask]

        filtered = (
            self.temporal_alpha * prev
            + (1.0 - self.temporal_alpha) * stable_curve
        )

        self.prev_filtered_curve = filtered.copy()
        self.point_history.append(filtered.copy())

        stacked = np.stack(list(self.point_history), axis=0)
        smoothed = np.mean(stacked, axis=0)

        return smoothed.astype(np.float32)


    def temporal_filter_ordered_raw_points(self, points):
        points = np.asarray(points, dtype=np.float32)

        if points.shape[0] == 0:
            return points

        if self.prev_filtered_raw_points is None:
            self.prev_filtered_raw_points = points.copy()
            self.raw_point_history.append(points.copy())
            return points

        if self.prev_filtered_raw_points.shape != points.shape:
            self.prev_filtered_raw_points = points.copy()
            self.raw_point_history.clear()
            self.raw_point_history.append(points.copy())
            return points

        prev = self.prev_filtered_raw_points

        forward_cost = np.mean(np.linalg.norm(points - prev, axis=1))
        reverse_cost = np.mean(np.linalg.norm(points[::-1] - prev, axis=1))

        if reverse_cost < forward_cost:
            points = points[::-1]

        disp = np.linalg.norm(points - prev, axis=1)
        stable_points = points.copy()

        jump_mask = disp > self.raw_max_interframe_motion
        stable_points[jump_mask] = prev[jump_mask]

        filtered = (
            self.raw_temporal_alpha * prev
            + (1.0 - self.raw_temporal_alpha) * stable_points
        )

        self.prev_filtered_raw_points = filtered.copy()
        self.raw_point_history.append(filtered.copy())

        stacked = np.stack(list(self.raw_point_history), axis=0)
        return np.mean(stacked, axis=0).astype(np.float32)

    def remove_large_neighbor_jumps(self, points, max_jump=0.12):
        if points.shape[0] < 3:
            return points

        kept = [points[0]]

        for p in points[1:]:
            if np.linalg.norm(p - kept[-1]) <= max_jump:
                kept.append(p)

        return np.asarray(kept, dtype=np.float32)

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

    def remove_near_duplicates(self, points, min_dist=1e-4):
        if points.shape[0] <= 1:
            return points

        kept = []

        for p in points:
            if len(kept) == 0:
                kept.append(p)
                continue

            dists = np.linalg.norm(np.asarray(kept) - p[None, :], axis=1)

            if np.min(dists) > min_dist:
                kept.append(p)

        return np.asarray(kept, dtype=np.float32)

    def order_points_smooth_open_curve(self, points):
        """
        Orders unordered rope points into one smooth open curve.

        Assumes:
          - one rope
          - smooth curve
          - no loop/self-intersection
        """
        points = np.asarray(points, dtype=np.float32)
        points = self.remove_near_duplicates(points)

        n = points.shape[0]

        if n <= 2:
            return points

        # Use farthest pair as candidate endpoints
        dmat = np.linalg.norm(
            points[:, None, :] - points[None, :, :],
            axis=2,
        )

        start_idx, end_idx = np.unravel_index(np.argmax(dmat), dmat.shape)

        def greedy_path(start):
            unused = set(range(n))
            order = [start]
            unused.remove(start)
            current_idx = start

            while unused:
                current = points[current_idx]
                unused_list = list(unused)
                candidates = points[unused_list]

                dists = np.linalg.norm(candidates - current[None, :], axis=1)
                nearest_idx = unused_list[int(np.argmin(dists))]

                order.append(nearest_idx)
                unused.remove(nearest_idx)
                current_idx = nearest_idx

            return order

        def path_cost(order):
            pts = points[order]
            segs = np.diff(pts, axis=0)
            seg_lens = np.linalg.norm(segs, axis=1)

            length_cost = np.sum(seg_lens)

            if len(segs) < 2:
                return length_cost

            unit = segs / (seg_lens[:, None] + 1e-8)
            turn_cost = np.sum(np.linalg.norm(np.diff(unit, axis=0), axis=1))

            return length_cost + 0.25 * turn_cost

        order_a = greedy_path(start_idx)
        order_b = greedy_path(end_idx)

        if path_cost(order_b) < path_cost(order_a):
            best_order = order_b
        else:
            best_order = order_a

        return points[best_order].astype(np.float32)

    def fit_rope_spline(
        self,
        P,
        Q_prime,
        is_ordered=False,
        num_output_points=300,
        smoothing=0.05,
    ):
        P = np.asarray(P, dtype=np.float32)
        Q_prime = np.asarray(Q_prime, dtype=np.float32)

        if P.shape[0] == 0 and Q_prime.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        if P.shape[0] == 0:
            P_final = Q_prime
        elif Q_prime.shape[0] == 0:
            P_final = P
        else:
            if is_ordered and P.shape == Q_prime.shape:
                P_final = (P + Q_prime) / 2.0
            else:
                P_final = np.vstack((P, Q_prime))

        P_final = self.order_points_smooth_open_curve(P_final)
        P_final = self.remove_large_neighbor_jumps(P_final, max_jump=0.12)
        P_final = self.temporal_filter_ordered_raw_points(P_final)

        if P_final.shape[0] < 2:
            return P_final

        if P_final.shape[0] == 2:
            t = np.linspace(0.0, 1.0, num_output_points)
            rope_curve = (
                (1.0 - t[:, None]) * P_final[0]
                + t[:, None] * P_final[1]
            )
            return rope_curve.astype(np.float32)

        data_T = P_final.T
        k = min(3, P_final.shape[0] - 1)

        try:
            tck, _ = splprep(
                data_T,
                u=None,
                k=k,
                s=smoothing,
            )

            u_fine = np.linspace(0.0, 1.0, num_output_points)
            spline_points = splev(u_fine, tck)

            rope_curve = np.array(spline_points).T
            return rope_curve.astype(np.float32)

        except Exception as e:
            self.get_logger().warn(f"Spline fit failed, using ordered raw points: {e}")
            return P_final.astype(np.float32)

    def synced_callback(self, cam1_msg: PointCloud2, cam2_msg: PointCloud2):
        P = self.pointcloud_to_numpy(cam1_msg)
        Q_prime = self.pointcloud_to_numpy(cam2_msg)

        rope_curve = self.fit_rope_spline(
            P,
            Q_prime,
            is_ordered=self.is_ordered,
            num_output_points=self.num_output_points,
            smoothing=self.smoothing,
        )

        rope_curve = self.temporal_filter_curve(rope_curve)

        if rope_curve.shape[0] == 0:
            self.get_logger().warn("No valid rope points received")
            return

        header = cam1_msg.header

        out_msg = pc2.create_cloud_xyz32(
            header,
            rope_curve.tolist(),
        )

        self.pub.publish(out_msg)

        self.get_logger().info(
            f"cam1 pts={P.shape[0]}, "
            f"cam2 pts={Q_prime.shape[0]}, "
            f"spline pts={rope_curve.shape[0]}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = TwoCloudRopeSplineFitter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()