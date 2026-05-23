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
        self.declare_parameter(
            "end_effector_pose_topic",
            "/right/workstation/end_effector_pose",
        )
        self.declare_parameter("save_path", "saved_rope_state.npy")

        self.rope_poses_topic = str(self.get_parameter("rope_poses_topic").value)
        self.num_segments = int(self.get_parameter("num_segments").value)
        self.rope_diameter = float(self.get_parameter("rope_diameter").value)
        self.end_effector_pose_topic = str(
            self.get_parameter("end_effector_pose_topic").value
        )
        self.save_path = str(self.get_parameter("save_path").value)

        self.latest_points = None
        self.latest_ee_pos = None
        self.lock = threading.Lock()

        self.sub = self.create_subscription(
            PoseArray,
            self.rope_poses_topic,
            self.rope_poses_callback,
            10,
        )

        self.ee_sub = self.create_subscription(
            PoseStamped,
            self.end_effector_pose_topic,
            self.end_effector_pose_callback,
            10,
        )

        self.get_logger().info(f"Subscribed to {self.rope_poses_topic}")
        self.get_logger().info(f"Subscribed to {self.end_effector_pose_topic}")
        self.get_logger().info(f"Will save state once to {self.save_path}")

    def end_effector_pose_callback(self, msg):
        p = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float32,
        )

        with self.lock:
            self.latest_ee_pos = p

    def get_latest_ee_pos(self):
        with self.lock:
            if self.latest_ee_pos is None:
                return None
            return self.latest_ee_pos.copy()

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

        self.get_logger().info(f"Received {len(points)} rope poses")

    def get_latest_points(self):
        with self.lock:
            if self.latest_points is None:
                return None
            return self.latest_points.copy()


def state_from_rope_points(
    env,
    state,
    sampled_points,
    grip_position=None,
):
    sampled_points = jnp.asarray(sampled_points)

    x_node, x_weld, x_grip = env.unpack_state(state)

    if sampled_points.shape != x_node.shape:
        raise ValueError(
            f"sampled_points shape {sampled_points.shape} does not match "
            f"x_node shape {x_node.shape}"
        )

    x_node = sampled_points

    if grip_position is not None:
        grip_position = jnp.asarray(grip_position)

        if grip_position.shape == (3,):
            grip_position = grip_position[None, :]

        if grip_position.shape != x_grip.shape:
            raise ValueError(
                f"grip_position shape {grip_position.shape} does not match "
                f"x_grip shape {x_grip.shape}"
            )

        x_grip = grip_position

    new_state = jnp.concatenate(
        [
            x_node.reshape(-1),
            x_weld.reshape(-1),
            x_grip.reshape(-1),
        ]
    )

    return new_state


def resample_fixed_link_length_extend(points, num_points, link_length=0.1):
    points = np.asarray(points, dtype=np.float32)

    if points.shape[0] < 2:
        return None

    sampled = [points[0].copy()]
    last = points[0].astype(np.float32)

    i = 1
    last_dir = None

    while i < len(points) and len(sampled) < num_points:
        p = points[i]
        v = p - last
        dist = np.linalg.norm(v)

        if dist < 1e-8:
            i += 1
            continue

        direction = v / dist
        last_dir = direction

        if dist >= link_length:
            new_p = last + link_length * direction
            sampled.append(new_p.astype(np.float32))
            last = new_p.astype(np.float32)
            # do not increment i
        else:
            i += 1

    # If observed rope ended early, extend straight using last direction
    if len(sampled) < num_points:
        if last_dir is None:
            v = points[-1] - points[-2]
            norm = np.linalg.norm(v)
            if norm < 1e-8:
                return None
            last_dir = v / norm

        while len(sampled) < num_points:
            new_p = sampled[-1] + link_length * last_dir
            sampled.append(new_p.astype(np.float32))

    return np.asarray(sampled, dtype=np.float32)


def save_state_once(
    node,
    save_path,
    state,
    rope_points,
    sampled_rope_points,
    grip_position,
    already_saved,
):
    if already_saved:
        return True

    if rope_points is None:
        return False

    if sampled_rope_points is None:
        return False

    if grip_position is None:
        return False

    save_data = {
        "state": np.asarray(state),
        "rope_points_raw": np.asarray(rope_points),
        "rope_points_sampled": np.asarray(sampled_rope_points),
        "grip_position": np.asarray(grip_position),
        "timestamp": time.time(),
    }

    np.save(save_path, save_data, allow_pickle=True)

    node.get_logger().info(f"Saved current rope/gripper state to {save_path}")

    return True


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

    env = RopeEnv(
        time_step=0.02,
        num_segments=10,
        rope_length=1.0144948,
        rope_diameter=0.01,
        youngs_modulus=1e5,
        mass_density=300,
        num_floating_grippers=1,
        grip_stiffness=300,
        gripper_radius=0.05,
        contact_smoothing=3e-3,
    )

    x_grip = jnp.array([[0.05, 0.2 / jnp.pi, 0.002]])
    state = env.state(x_grip=x_grip)

    env.visualize(server, state)

    saved_state = False

    try:
        while rclpy.ok():
            start = time.time()

            ee_pos = node.get_latest_ee_pos()

            if ee_pos is not None:
                x_node, x_weld, x_grip = env.unpack_state(state)

                ee_grip = jnp.asarray(ee_pos)

                if ee_grip.shape == (3,):
                    ee_grip = ee_grip[None, :]

                state = jnp.concatenate(
                    [
                        x_node.reshape(-1),
                        x_weld.reshape(-1),
                        ee_grip.reshape(-1),
                    ]
                )

            points = node.get_latest_points()
            sampled = None

            if points is not None and points.shape[0] >= 2:
                sampled = resample_fixed_link_length_extend(
                    points,
                    node.num_segments + 1,
                    link_length=0.1,
                )

                if sampled is not None:
                    sampled_jnp = jnp.asarray(sampled)
                    # print("State shape:", state.shape)
                    state = state_from_rope_points(
                        env,
                        state,
                        sampled_jnp,
                        grip_position=ee_pos,
                    )

            saved_state = save_state_once(
                node=node,
                save_path=node.save_path,
                state=state,
                rope_points=points,
                sampled_rope_points=sampled,
                grip_position=ee_pos,
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