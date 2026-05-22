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
from geometry_msgs.msg import PoseArray

from pathlib import Path
import sys

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
        self.declare_parameter("num_segments", 11)
        self.declare_parameter("rope_diameter", 0.01)

        self.rope_poses_topic = str(self.get_parameter("rope_poses_topic").value)
        self.num_segments = int(self.get_parameter("num_segments").value)
        self.rope_diameter = float(self.get_parameter("rope_diameter").value)

        self.latest_points = None
        self.lock = threading.Lock()

        self.sub = self.create_subscription(
            PoseArray,
            self.rope_poses_topic,
            self.rope_poses_callback,
            10,
        )

        self.get_logger().info(f"Subscribed to {self.rope_poses_topic}")

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

        # Pad by repeating last point until we have at least 10
        if points.shape[0] < 10:
            last_point = points[-1:]

            repeat_count = 10 - points.shape[0]

            padding = np.repeat(
                last_point,
                repeat_count,
                axis=0,
            )

            points = np.concatenate(
                [points, padding],
                axis=0,
            )

            self.get_logger().warn(
                f"Padded rope poses from "
                f"{points.shape[0] - repeat_count} to 10"
            )

        with self.lock:
            self.latest_points = points

        self.get_logger().info(
            f"Received {len(points)} rope poses"
        )

    def get_latest_points(self):
        with self.lock:
            if self.latest_points is None:
                return None
            return self.latest_points.copy()

def state_from_rope_points(env, state, sampled_points):
    sampled_points = jnp.asarray(sampled_points)

    x_node, x_weld, x_grip = env.unpack_state(state)

    if sampled_points.shape != x_node.shape:
        raise ValueError(
            f"sampled_points shape {sampled_points.shape} does not match "
            f"x_node shape {x_node.shape}"
        )

    x_node = sampled_points

    new_state = jnp.concatenate(
        [
            x_node.reshape(-1),
            x_weld.reshape(-1),
            x_grip.reshape(-1),
        ]
    )

    return new_state

def resample_points(points, num_points):
    points = np.asarray(points, dtype=np.float32)

    if points.shape[0] < 2:
        return None

    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)

    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]

    if total_length < 1e-6:
        return None

    target = np.linspace(0.0, total_length, num_points)

    sampled = np.zeros((num_points, 3), dtype=np.float32)
    sampled[:, 0] = np.interp(target, cumulative, points[:, 0])
    sampled[:, 1] = np.interp(target, cumulative, points[:, 1])
    sampled[:, 2] = np.interp(target, cumulative, points[:, 2])

    return sampled


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
        num_segments=node.num_segments,
        rope_length=1.57,
        rope_diameter=node.rope_diameter,
        youngs_modulus=1e5,
        mass_density=300,
        num_floating_grippers=1,
        grip_stiffness=100,
    )

    x_grip = jnp.array([[0.05, 0.2 / jnp.pi, 0.002]])
    state = env.state(x_grip=x_grip)

    env.visualize(server, state)

    rope_handle = None

    try:
        while rclpy.ok():
            start = time.time()

            points = node.get_latest_points()

            if points is not None and points.shape[0] >= 2:
                sampled = resample_points(
                    points,
                    node.num_segments + 1,
                )

                if sampled is not None:
                    sampled_jnp = jnp.asarray(sampled)

                    if hasattr(env, "state_from_rope_points") or True:
                        state = state_from_rope_points(
                            env,
                            state,
                            sampled_jnp,
                        )
                        env.visualize(server, state)
                    else:
                        if rope_handle is not None:
                            rope_handle.remove()

                        rope_handle = server.scene.add_spline_catmull_rom(
                            name="/measured_rope",
                            positions=sampled,
                            color=(255, 80, 80),
                            line_width=6.0,
                        )

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