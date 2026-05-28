#!/usr/bin/env python3

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

from environments import ClothEnv


class ClothGridRenderer(Node):
    def __init__(self):
        super().__init__("cloth_grid_renderer")

        self.declare_parameter(
            "cloth_grid_topic",
            "/front/sam_cloth/cloth_grid_poses",
        )

        self.cloth_grid_topic = str(
            self.get_parameter("cloth_grid_topic").value
        )

        self.lock = threading.Lock()

        self.latest_points = None
        self.latest_left_grip = None
        self.latest_right_grip = None

        self.sub = self.create_subscription(
            PoseArray,
            self.cloth_grid_topic,
            self.cloth_callback,
            10,
        )

        self.left_ee_sub = self.create_subscription(
            PoseStamped,
            "/left/workstation/end_effector_pose",
            self.left_ee_callback,
            10,
        )

        self.right_ee_sub = self.create_subscription(
            PoseStamped,
            "/right/workstation/end_effector_pose",
            self.right_ee_callback,
            10,
        )

        self.get_logger().info(f"Subscribed to {self.cloth_grid_topic}")
        self.get_logger().info("Subscribed to /left/workstation/end_effector_pose")
        self.get_logger().info("Subscribed to /right/workstation/end_effector_pose")

    def cloth_callback(self, msg):
        if len(msg.poses) == 0:
            return

        points = np.asarray(
            [
                [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                ]
                for pose in msg.poses
            ],
            dtype=np.float32,
        )

        with self.lock:
            self.latest_points = points

        self.get_logger().info(
            f"Received {points.shape[0]} cloth points"
        )

    def left_ee_callback(self, msg):
        p = msg.pose.position

        with self.lock:
            self.latest_left_grip = np.asarray(
                [p.x, p.y, p.z],
                dtype=np.float32,
            )

    def right_ee_callback(self, msg):
        p = msg.pose.position

        with self.lock:
            self.latest_right_grip = np.asarray(
                [p.x, p.y, p.z],
                dtype=np.float32,
            )

    def get_latest_points(self):
        with self.lock:
            if self.latest_points is None:
                return None

            return self.latest_points.copy()

    def get_latest_grip_positions(self):
        with self.lock:
            if self.latest_left_grip is None or self.latest_right_grip is None:
                return None

            return np.stack(
                [
                    self.latest_left_grip,
                    self.latest_right_grip,
                ],
                axis=0,
            ).copy()


def state_from_cloth_points(
    env,
    state,
    sampled_points,
    grip_positions=None,
):
    sampled_points = jnp.asarray(sampled_points)

    x_node, x_grip = env.unpack_state(state)

    if sampled_points.shape != x_node.shape:
        raise ValueError(
            f"sampled_points shape {sampled_points.shape} "
            f"does not match x_node shape {x_node.shape}"
        )

    x_node = sampled_points

    if grip_positions is not None:
        grip_positions = jnp.asarray(grip_positions)

        if grip_positions.shape != x_grip.shape:
            raise ValueError(
                f"grip_positions shape {grip_positions.shape} "
                f"does not match x_grip shape {x_grip.shape}"
            )

        x_grip = grip_positions

    new_state = jnp.concatenate(
        [
            x_node.reshape(-1),
            x_grip.reshape(-1),
        ]
    )

    return new_state


def check_node_distances(points, num_div_side):
    pts = np.asarray(points, dtype=np.float32).reshape(
        num_div_side,
        num_div_side,
        3,
    )

    row_dists = np.linalg.norm(
        pts[:, 1:, :] - pts[:, :-1, :],
        axis=-1,
    )

    col_dists = np.linalg.norm(
        pts[1:, :, :] - pts[:-1, :, :],
        axis=-1,
    )

    print("\n========== Cloth node distance check ==========")
    print(
        f"row neighbor distances: "
        f"min={row_dists.min():.6f}, "
        f"max={row_dists.max():.6f}, "
        f"mean={row_dists.mean():.6f}"
    )

    print(
        f"col neighbor distances: "
        f"min={col_dists.min():.6f}, "
        f"max={col_dists.max():.6f}, "
        f"mean={col_dists.mean():.6f}"
    )

    print("\nRow distances:")
    print(row_dists)

    print("\nColumn distances:")
    print(col_dists)

    print("==============================================\n")


NUM_SIM_STEPS_PER_FRAME = 5


def main(args=None):
    rclpy.init(args=args)

    node = ClothGridRenderer()

    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True,
    )
    spin_thread.start()

    server = viser.ViserServer()
    _ = server.scene.add_grid(
        name="ground",
        position=(0, 0, -0.009),
    )

    env = ClothEnv.from_regular_grid(
        time_step=0.02,
        cloth_width=0.38,
        num_div_side=9,
        thickness=0.5e-3,
        youngs_modulus=1e4,
        possion_ratio=0.3,
        mass_density=20,
        num_floating_grippers=2,
        grip_stiffness=1000,
        gripper_radius=0.01,
        contact_smoothing=3e-3,
        ground_friction_coeff=0.8,
    )

    state = env.state()

    env.visualize(server, state)

    try:
        while rclpy.ok():
            start = time.time()

            points = node.get_latest_points()
            grip_positions = node.get_latest_grip_positions()

            if points is not None:
                if points.shape[0] != env.params.num_nodes:
                    node.get_logger().warn(
                        f"Expected {env.params.num_nodes} cloth nodes, "
                        f"got {points.shape[0]}. Skipping."
                    )
                else:
                    # Optional debugging:
                    # check_node_distances(points, num_div_side=9)

                    state = state_from_cloth_points(
                        env=env,
                        state=state,
                        sampled_points=points,
                        grip_positions=grip_positions,
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