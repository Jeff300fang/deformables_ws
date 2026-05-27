#!/usr/bin/env python3

import os
import sys
import time
import threading
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import rclpy
import viser

from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped


JAX_DEFORMABLES_PATH = Path(
    "/home/jeff/trustworthroboticsgroup/CoRL2026/deformables_ws/src/jax-deformable"
)

if not JAX_DEFORMABLES_PATH.exists():
    raise RuntimeError(f"jax-deformables path not found: {JAX_DEFORMABLES_PATH}")

sys.path.insert(0, str(JAX_DEFORMABLES_PATH))

from environments import RopeEnv


class RopePoseSimRenderer(Node):
    def __init__(self):
        super().__init__("rope_pose_sim_renderer")

        self.declare_parameter("rope_poses_topic", "/rope_poses")
        self.declare_parameter("num_segments", 10)
        self.declare_parameter("rope_diameter", 0.01)
        self.declare_parameter("save_path", "saved_rope_state.npy")

        self.rope_poses_topic = str(self.get_parameter("rope_poses_topic").value)
        self.num_segments = int(self.get_parameter("num_segments").value)
        self.rope_diameter = float(self.get_parameter("rope_diameter").value)
        self.save_path = str(self.get_parameter("save_path").value)

        self.declare_parameter("parabola_obstacle_enabled", True)
        self.declare_parameter("parabola_obstacle_x_center", 0.125)
        self.declare_parameter("parabola_obstacle_z_base", 0.0)
        self.declare_parameter("parabola_obstacle_height", 0.24)
        self.declare_parameter("parabola_obstacle_half_width", 0.045)
        self.declare_parameter("parabola_obstacle_clearance", 0.012)

        self.parabola_obstacle_enabled = bool(self.get_parameter("parabola_obstacle_enabled").value)
        self.parabola_obstacle_x_center = float(self.get_parameter("parabola_obstacle_x_center").value)
        self.parabola_obstacle_z_base = float(self.get_parameter("parabola_obstacle_z_base").value)
        self.parabola_obstacle_height = float(self.get_parameter("parabola_obstacle_height").value)
        self.parabola_obstacle_half_width = float(self.get_parameter("parabola_obstacle_half_width").value)
        self.parabola_obstacle_clearance = float(self.get_parameter("parabola_obstacle_clearance").value)

        self.lock = threading.Lock()

        self.latest_points = None
        self.latest_ee_pos_left = None
        self.latest_ee_pos_right = None

        self.sub = self.create_subscription(
            PoseArray,
            self.rope_poses_topic,
            self.rope_poses_callback,
            10,
        )

        self.ee_sub_right = self.create_subscription(
            PoseStamped,
            "/right/workstation/end_effector_pose",
            self.end_effector_right_pose_callback,
            10,
        )

        self.ee_sub_left = self.create_subscription(
            PoseStamped,
            "/left/workstation/end_effector_pose",
            self.end_effector_left_pose_callback,
            10,
        )

        self.declare_parameter("in_contact", True)

        self.in_contact = bool(
            self.get_parameter("in_contact").value
        )

    def end_effector_right_pose_callback(self, msg):
        p = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float32,
        )

        with self.lock:
            self.latest_ee_pos_right = p

    def end_effector_left_pose_callback(self, msg):
        p = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float32,
        )

        with self.lock:
            self.latest_ee_pos_left = p

    def get_latest_points(self):
        with self.lock:
            if self.latest_points is None:
                return None
            return self.latest_points.copy()

    def get_latest_grip_positions(self):
        with self.lock:
            if (
                self.latest_ee_pos_left is None
                or self.latest_ee_pos_right is None
            ):
                return None

            return np.stack(
                [
                    self.latest_ee_pos_left,
                    self.latest_ee_pos_right,
                ],
                axis=0,
            )

    def rope_poses_callback(self, msg):
        if len(msg.poses) == 0:
            return

        points = []

        for pose in msg.poses:
            points.append(
                [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                ]
            )

        points = np.asarray(points, dtype=np.float32)

        if points.shape[0] < 10:
            last_point = points[-1:]
            repeat_count = 10 - points.shape[0]
            padding = np.repeat(last_point, repeat_count, axis=0)
            points = np.concatenate([points, padding], axis=0)

            self.get_logger().warn(
                f"Padded rope poses from {points.shape[0] - repeat_count} to 10"
            )

        with self.lock:
            self.latest_points = points
        self.get_logger().info(f"{points}")
        self.get_logger().info(f"Received {len(points)} rope poses")


def state_from_rope_points(
    env,
    state,
    sampled_points,
    grip_positions=None,
):
    sampled_points = jnp.asarray(sampled_points)

    x_node, x_weld, x_grip = env.unpack_state(state)

    if sampled_points.shape != x_node.shape:
        raise ValueError(
            f"sampled_points shape {sampled_points.shape} does not match "
            f"x_node shape {x_node.shape}"
        )

    x_node = sampled_points

    if grip_positions is not None:
        grip_positions = jnp.asarray(grip_positions)

        if grip_positions.shape != x_grip.shape:
            raise ValueError(
                f"grip_positions shape {grip_positions.shape} does not match "
                f"x_grip shape {x_grip.shape}"
            )

        x_grip = grip_positions

    new_state = jnp.concatenate(
        [
            x_node.reshape(-1),
            x_weld.reshape(-1),
            x_grip.reshape(-1),
        ]
    )

    return new_state


def resample_fixed_link_length_extend_lr(
    points,
    num_points,
    link_length=0.1,
    grip_positions=None,
):
    points = np.asarray(points, dtype=np.float32)

    if points.shape[0] < 2:
        return None

    # grip_positions is expected as [left_grip, right_grip]
    if grip_positions is not None:
        grip_positions = np.asarray(grip_positions, dtype=np.float32)
        left_grip = grip_positions[0]
        right_grip = grip_positions[1]

        # Make point order deterministic:
        # points[0] should be the rope end closer to the left gripper.
        d_start_left = np.linalg.norm(points[0] - left_grip)
        d_end_left = np.linalg.norm(points[-1] - left_grip)

        if d_end_left < d_start_left:
            points = points[::-1].copy()

    # First do normal fixed-link resampling from left to right.
    sampled = [points[0].copy()]
    last = points[0].astype(np.float32)

    i = 1
    while i < len(points) and len(sampled) < num_points:
        p = points[i]
        v = p - last
        dist = np.linalg.norm(v)

        if dist < 1e-8:
            i += 1
            continue

        direction = v / dist

        if dist >= link_length:
            new_p = last + link_length * direction
            sampled.append(new_p.astype(np.float32))
            last = new_p.astype(np.float32)
        else:
            i += 1

    if len(sampled) < 2:
        return None

    # Extend deterministically:
    # left, right, left, right, ...
    extend_left_next = True

    while len(sampled) < num_points:
        if extend_left_next:
            # Outward direction from second node toward first node.
            v = sampled[0] - sampled[1]
            norm = np.linalg.norm(v)

            if norm < 1e-8:
                return None

            direction = v / norm
            new_p = sampled[0] + link_length * direction
            sampled.insert(0, new_p.astype(np.float32))

        else:
            # Outward direction from second-last node toward last node.
            v = sampled[-1] - sampled[-2]
            norm = np.linalg.norm(v)

            if norm < 1e-8:
                return None

            direction = v / norm
            new_p = sampled[-1] + link_length * direction
            sampled.append(new_p.astype(np.float32))

        extend_left_next = not extend_left_next

    return np.asarray(sampled, dtype=np.float32)


def save_state_once(
    node,
    save_path,
    state,
    rope_points,
    sampled_rope_points,
    grip_positions,
    already_saved,
):
    if already_saved:
        return True

    if rope_points is None:
        return False

    if sampled_rope_points is None:
        return False

    if grip_positions is None:
        return False

    save_data = {
        "state": np.asarray(state),
        "rope_points_raw": np.asarray(rope_points),
        "rope_points_sampled": np.asarray(sampled_rope_points),
        "grip_positions": np.asarray(grip_positions),
        "timestamp": time.time(),
    }

    np.save(save_path, save_data, allow_pickle=True)

    node.get_logger().info(f"Saved current rope/gripper state to {save_path}")

    return True


def resample_fixed_spacing(points, num_points, spacing=0.1):
    pts = np.asarray(points, dtype=np.float32)

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)

    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = s[-1]

    # exactly num_points samples
    new_s = np.arange(num_points, dtype=np.float32) * spacing

    new_pts = []
    j = 0

    for target in new_s:
        if target <= total_len:
            while j < len(s) - 2 and s[j + 1] < target:
                j += 1

            denom = max(s[j + 1] - s[j], 1e-8)
            t = (target - s[j]) / denom
            p = (1 - t) * pts[j] + t * pts[j + 1]
        else:
            # extend past observed curve if needed
            direction = pts[-1] - pts[-2]
            direction = direction / max(np.linalg.norm(direction), 1e-8)
            p = pts[-1] + (target - total_len) * direction

        new_pts.append(p.astype(np.float32))

    return np.asarray(new_pts, dtype=np.float32)

def teleport_rope_closest_points_shear(sampled_points, grip_positions):
    pts = np.asarray(sampled_points, dtype=np.float32).copy()
    grips = np.asarray(grip_positions, dtype=np.float32)

    # arc-length parameter along rope
    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    s_nodes = np.concatenate([[0.0], np.cumsum(seg_len)])

    anchors_s = []
    anchors_delta = []

    for g in grips:
        # closest point on rope segments
        a = pts[:-1]
        b = pts[1:]
        ab = b - a
        ab_len2 = np.sum(ab * ab, axis=1)

        t = np.sum((g[None, :] - a) * ab, axis=1) / np.maximum(ab_len2, 1e-12)
        t = np.clip(t, 0.0, 1.0)

        closest = a + t[:, None] * ab
        d = np.linalg.norm(closest - g[None, :], axis=1)

        k = int(np.argmin(d))
        s_anchor = s_nodes[k] + t[k] * seg_len[k]
        delta = g - closest[k]

        anchors_s.append(s_anchor)
        anchors_delta.append(delta)

    s0, s1 = anchors_s
    d0, d1 = anchors_delta

    if abs(s1 - s0) < 1e-8:
        return pts

    # Linear shear field along rope arc length.
    # This guarantees:
    #   rope(s0) -> left gripper
    #   rope(s1) -> right gripper
    w1 = (s_nodes - s0) / (s1 - s0)
    w0 = 1.0 - w1

    displacement = w0[:, None] * d0[None, :] + w1[:, None] * d1[None, :]
    return (pts + displacement).astype(np.float32)

def project_grippers_to_nearest_rope_points(sampled_points, grip_positions):
    pts = np.asarray(sampled_points, dtype=np.float32)
    grips = np.asarray(grip_positions, dtype=np.float32).copy()

    new_grips = []

    for grip in grips:
        dists = np.linalg.norm(pts - grip[None, :], axis=1)
        idx = int(np.argmin(dists))
        new_grips.append(pts[idx])

    return np.asarray(new_grips, dtype=np.float32)

def teleport_rope_closest_points_rigid(sampled_points, grip_positions):
    sampled_points = np.asarray(sampled_points, dtype=np.float32)
    grip_positions = np.asarray(grip_positions, dtype=np.float32)

    if sampled_points is None or grip_positions is None:
        return sampled_points

    # closest rope node to each gripper
    closest_idxs = []
    closest_points = []

    for grip_pos in grip_positions:
        dists = np.linalg.norm(sampled_points - grip_pos[None, :], axis=1)
        idx = int(np.argmin(dists))
        closest_idxs.append(idx)
        closest_points.append(sampled_points[idx])

    src = np.asarray(closest_points, dtype=np.float32)
    dst = grip_positions.astype(np.float32)

    src_center = np.mean(src, axis=0)
    dst_center = np.mean(dst, axis=0)

    src_vec = src[1] - src[0]
    dst_vec = dst[1] - dst[0]

    src_norm = np.linalg.norm(src_vec)
    dst_norm = np.linalg.norm(dst_vec)

    if src_norm < 1e-8 or dst_norm < 1e-8:
        # fallback: translation only
        offset = dst_center - src_center
        return (sampled_points + offset).astype(np.float32)

    a = src_vec / src_norm
    b = dst_vec / dst_norm

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-8:
        if c > 0:
            R = np.eye(3, dtype=np.float32)
        else:
            # 180 degree rotation around any axis perpendicular to a
            tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(np.dot(tmp, a)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            axis = np.cross(a, tmp)
            axis = axis / np.linalg.norm(axis)

            K = np.array(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ],
                dtype=np.float32,
            )

            R = np.eye(3, dtype=np.float32) + 2.0 * (K @ K)
    else:
        K = np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ],
            dtype=np.float32,
        )

        R = (
            np.eye(3, dtype=np.float32)
            + K
            + K @ K * ((1.0 - c) / (s**2))
        )

    teleported = (sampled_points - src_center) @ R.T + dst_center
    return teleported.astype(np.float32)

def parabola_obstacle_z(x, x_center, z_base, height, half_width):
    u = (x - x_center) / max(half_width, 1e-8)
    return z_base + height * (1.0 - u * u)


def main(args=None):
    rclpy.init(args=args)

    node = RopePoseSimRenderer()

    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True,
    )
    spin_thread.start()

    server = viser.ViserServer()
    _ = server.scene.add_grid(name="ground")

    if node.parabola_obstacle_enabled:
        xs = np.linspace(
            node.parabola_obstacle_x_center - node.parabola_obstacle_half_width,
            node.parabola_obstacle_x_center + node.parabola_obstacle_half_width,
            80,
        )
        ys = np.linspace(-0.35, 0.35, 8)

        obstacle_pts = []
        for y in ys:
            for x in xs:
                z = parabola_obstacle_z(
                    x,
                    node.parabola_obstacle_x_center,
                    node.parabola_obstacle_z_base,
                    node.parabola_obstacle_height,
                    node.parabola_obstacle_half_width,
                )
                obstacle_pts.append([x, y, z])

        obstacle_pts = np.asarray(obstacle_pts, dtype=np.float32)

        server.scene.add_point_cloud(
            name="/parabola_obstacle",
            points=obstacle_pts,
            colors=np.tile(np.array([[255, 80, 80]], dtype=np.uint8), (obstacle_pts.shape[0], 1)),
            point_size=0.01,
        )

    env = RopeEnv(
        time_step=0.02,
        num_segments=10,
        rope_length=1.0144948,
        rope_diameter=0.01,
        youngs_modulus=1e5,
        mass_density=300,
        num_floating_grippers=2,
        grip_stiffness=300,
        gripper_radius=0.05,
        contact_smoothing=3e-3,
    )

    x_grip = jnp.array(
        [
            [0.05, -0.10, 0.002],
            [0.05,  0.10, 0.002],
        ]
    )

    state = env.state(x_grip=x_grip)

    env.visualize(server, state)

    saved_state = False

    try:
        while rclpy.ok():
            start = time.time()

            grip_positions = node.get_latest_grip_positions()

            points = node.get_latest_points()

            if grip_positions is not None and points is not None and points.shape[0] >= 2:
                pass
            elif grip_positions is not None:
                x_node, x_weld, x_grip = env.unpack_state(state)

                grip_positions_jnp = jnp.asarray(grip_positions)

                state = jnp.concatenate(
                    [
                        x_node.reshape(-1),
                        x_weld.reshape(-1),
                        grip_positions_jnp.reshape(-1),
                    ]
                )
            sampled = None

            if points is not None and points.shape[0] >= 2:
                sampled = resample_fixed_link_length_extend_lr(
                    points,
                    node.num_segments + 1,
                    link_length=0.1,
                    grip_positions=grip_positions,
                )

                if sampled is not None:
                    if node.in_contact and grip_positions is not None:
                        sampled = teleport_rope_closest_points_shear(
                            sampled_points=sampled,
                            grip_positions=grip_positions,
                        )


                        sampled = resample_fixed_spacing(
                            sampled,
                            num_points=env.params.num_nodes,
                            spacing=env.params.segment_length,
                        )
                        grip_positions = project_grippers_to_nearest_rope_points(
                            sampled_points=sampled,
                            grip_positions=grip_positions,
                        )
                                            
                    alpha = 0.7

                    if hasattr(node, "prev_sampled") and node.prev_sampled is not None:
                        sampled = alpha * node.prev_sampled + (1.0 - alpha) * sampled

                    node.prev_sampled = sampled.copy()
                    sampled_jnp = jnp.asarray(sampled)

                    state = state_from_rope_points(
                        env,
                        state,
                        sampled_jnp,
                        grip_positions=grip_positions,
                    )

            saved_state = save_state_once(
                node=node,
                save_path=node.save_path,
                state=state,
                rope_points=points,
                sampled_rope_points=sampled,
                grip_positions=grip_positions,
                already_saved=saved_state,
            )

            env.visualize(server, state)

            elapsed = time.time() - start
            wait = env.params.dt - elapsed

            if wait > 0:
                time.sleep(wait)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()