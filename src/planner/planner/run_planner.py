#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import rclpy
import viser

from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import Bool

JAX_DEFORMABLES_PATH = Path(
    "/home/jeff/trustworthroboticsgroup/CoRL2026/deformables_ws/src/jax-deformable"
)
sys.path.insert(0, str(JAX_DEFORMABLES_PATH))

from environments import RopeEnv
from planners import gpu_sls


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



def make_control_constraints(u_min, u_max):
    def constraints(x, u, t):
        return jnp.concatenate((u - u_max, u_min - u))

    return constraints


def make_constant_disturbance(alpha):
    def disturbance(X):
        N, nx = X.shape
        E0 = alpha * jnp.eye(nx, dtype=X.dtype)
        return jnp.broadcast_to(E0, (N, nx, nx))

    return disturbance


class RopeStateSolverNode(Node):
    def __init__(self):
        super().__init__("rope_state_solver_node")

        self.declare_parameter("rope_state_topic", "/rope_poses")
        self.rope_state_topic = str(self.get_parameter("rope_state_topic").value)

        self.latest_ee_pos_left = None
        self.latest_ee_pos_right = None

        self.latest_ee_pos_seq_left = 0
        self.latest_ee_pos_seq_right = 0

        self.ee_base_frame_seq_left = 0
        self.ee_base_frame_seq_right = 0

        self.last_solve_latest_ee_pos_seq_left = -1
        self.last_solve_latest_ee_pos_seq_right = -1

        self.last_solve_ee_base_frame_seq_left = -1
        self.last_solve_ee_base_frame_seq_right = -1

        self.server = viser.ViserServer()
        _ = self.server.scene.add_grid(name="ground")

        self.env = RopeEnv(
            time_step=0.02,
            num_segments=10,
            rope_length=1.0,
            rope_diameter=0.01,
            youngs_modulus=1e5,
            mass_density=300,
            num_floating_grippers=2,
            grip_stiffness=2000,
            gripper_radius=0.02,
            contact_smoothing=3e-3,
        )

        self.N = 50
        self.dt = self.env.params.dt

        self.x_grip = None
        self.state = None
        self.first_solve = True
        self.controller = None

        self.control0 = self.env.control(
            c_grip=jnp.array([1.0, 1.0]),
        )

        y_coords = (
            jnp.arange(self.env.params.num_nodes) * self.env.params.segment_length
            - 0.5
        )

        z_coords = jnp.ones(self.env.params.num_nodes) * 0.05

        nodes = jnp.stack(
            (
                jnp.zeros_like(y_coords),  # x
                y_coords,                  # y
                z_coords,                  # z
            ),
            axis=1,
        )

        self.state_goal = self.env.state(
            x_node=nodes,
        )

        vmax = 0.2
        u_max = jnp.array([vmax, vmax, vmax, 10.0])
        self.u_max = jnp.repeat(u_max, 2)

        self.constraints = make_control_constraints(
            u_min=-self.u_max,
            u_max=self.u_max,
        )

        self.sub = self.create_subscription(
            PoseArray,
            self.rope_state_topic,
            self.rope_state_callback,
            1,
        )

        self.goal_pose_pub_left = self.create_publisher(
            PoseStamped,
            "/left/iiwa/goal_pose",
            1,
        )

        self.goal_pose_pub_right = self.create_publisher(
            PoseStamped,
            "/right/iiwa/goal_pose",
            1,
        )

        self.ee_sub_left = self.create_subscription(
            PoseStamped,
            "/left/workstation/end_effector_pose",
            self.end_effector_left_pose_callback,
            1,
        )

        self.ee_sub_right = self.create_subscription(
            PoseStamped,
            "/right/workstation/end_effector_pose",
            self.end_effector_right_pose_callback,
            1,
        )

        self.ee_base_frame_sub_left = self.create_subscription(
            PoseStamped,
            "/left/end_effector_pose",
            self.ee_base_frame_left_callback,
            1,
        )

        self.ee_base_frame_sub_right = self.create_subscription(
            PoseStamped,
            "/right/end_effector_pose",
            self.ee_base_frame_right_callback,
            1,
        )

        self.grip_pub_left = self.create_publisher(
            Bool,
            "/left_grip",
            1,
        )

        self.grip_pub_right = self.create_publisher(
            Bool,
            "/right_grip",
            1,
        )

        self.get_logger().info(f"Subscribed to {self.rope_state_topic}")
        self.get_logger().info("Waiting for first real gripper pose before initializing state.")

        self.num_iterations = 0
        self.ee_base_frame_left = None
        self.ee_base_frame_right = None
        self.grasping_procedure = False
        self.gripper_closed = False
        self.grasing_starting_position = None

    def ee_base_frame_left_callback(self, msg):
        self.ee_base_frame_left = msg
        self.ee_base_frame_seq_left += 1

    def ee_base_frame_right_callback(self, msg):
        self.ee_base_frame_right = msg
        self.ee_base_frame_seq_right += 1

    def end_effector_left_pose_callback(self, msg):
        p = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float32,
        )

        self.latest_ee_pos_left = p
        self.latest_ee_pos_seq_left += 1

    def end_effector_right_pose_callback(self, msg):
        p = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float32,
        )

        self.latest_ee_pos_right = p
        self.latest_ee_pos_seq_right += 1

    def get_latest_grip_positions(self):
        if self.latest_ee_pos_left is None or self.latest_ee_pos_right is None:
            return None

        return np.stack(
            [
                self.latest_ee_pos_left,
                self.latest_ee_pos_right,
            ],
            axis=0,
        )

    def initialize_controller(self):
        def cost(W, reference, x, u, t):
            state_err = x - self.state_goal
            control_err = u - self.control0

            return (
                1.0 * jnp.sum(state_err[:-6] ** 2)
                + 0.1 * jnp.sum(control_err[:-2] ** 2)
            )

        def dynamics(x, u, t, parameter):
            return self.env.step(x, u)

        admm_cfg = gpu_sls.ADMMConfig(
            eps_abs=5e-2,
            eps_rel=1e-2,
            rho_max=1e3,
            max_iterations=100,
            rho_update_frequency=25,
            initial_rho=10.0,
        )

        sls_cfg = gpu_sls.SLSConfig(
            max_sls_iterations=2,
            sls_primal_tol=1e-2,
            enable_fastsls=False,
            initialize_nominal=True,
            max_initial_sqp_iterations=0,
            warm_start=False,
            rti=False,
        )

        sqp_cfg = gpu_sls.SQPConfig(
            max_sqp_iterations=1,
            warm_start=False,
            feas_tol=1e-2,
            step_tol=1e-4,
            line_search=True,
        )

        cfg = gpu_sls.MPCConfig(
            n=self.state.size,
            nu=self.control0.size,
            N=self.N,
            dt=self.dt,
            W=None,
            u_ref=self.control0,
        )

        disturbance = make_constant_disturbance(
            alpha=0.003 * self.dt,
        )

        nc = self.constraints(self.state, self.control0, 0.0).size

        self.controller = gpu_sls.GenericMPC(
            sls_cfg,
            sqp_cfg,
            admm_cfg,
            config=cfg,
            dynamics=dynamics,
            cost=cost,
            constraints=self.constraints,
            obstacles=jnp.zeros((0, 3)),
            disturbance=disturbance,
            num_constraints=nc,
            shift=1,
            X_in=None,
            U_in=None,
        )

        self.get_logger().info("Initialized MPC controller.")

    def closest_point_on_rope_segments(self, points, p):
        points = np.asarray(points, dtype=np.float32)
        p = np.asarray(p, dtype=np.float32)

        a = points[:-1]
        b = points[1:]
        ab = b - a

        ab_len2 = np.sum(ab * ab, axis=1)
        ap = p[None, :] - a

        t = np.sum(ap * ab, axis=1) / np.maximum(ab_len2, 1e-12)
        t = np.clip(t, 0.0, 1.0)

        closest = a + t[:, None] * ab
        dists = np.linalg.norm(closest - p[None, :], axis=1)

        seg_idx = int(np.argmin(dists))
        return float(dists[seg_idx]), seg_idx, float(t[seg_idx]), closest[seg_idx]

    def grasp(self):
        grip_positions = self.get_latest_grip_positions()

        if grip_positions is None:
            self.get_logger().warn("Cannot grasp: missing left/right workstation EE poses.")
            return

        if self.ee_base_frame_left is None or self.ee_base_frame_right is None:
            self.get_logger().warn("Cannot grasp: missing left/right base-frame EE poses.")
            return

        left_ee_pos = grip_positions[0]
        right_ee_pos = grip_positions[1]

        if self.grasing_starting_position is None:
            self.grasing_starting_position = {
                "left": self.ee_base_frame_left,
                "right": self.ee_base_frame_right,
            }

        left_at_ground = left_ee_pos[2] <= 0.005
        right_at_ground = right_ee_pos[2] <= 0.005

        if left_at_ground and right_at_ground and not self.gripper_closed:
            self.grip_pub_left.publish(Bool(data=True))
            self.grip_pub_right.publish(Bool(data=True))

            self.gripper_closed = True
            self.get_logger().info("Closing both grippers")
            time.sleep(3)
            self.get_logger().info("Finished closing both grippers")
            return

        if not self.gripper_closed:
            left_goal_pose = PoseStamped()
            left_goal_pose.header.stamp = self.get_clock().now().to_msg()
            left_goal_pose.header.frame_id = "world"
            left_goal_pose.pose.position.x = float(
                self.grasing_starting_position["left"].pose.position.x
            )
            left_goal_pose.pose.position.y = float(
                self.grasing_starting_position["left"].pose.position.y
            )
            left_goal_pose.pose.position.z = float(
                self.ee_base_frame_left.pose.position.z - 0.15 * self.dt
            )
            left_goal_pose.pose.orientation.w = 1.0

            right_goal_pose = PoseStamped()
            right_goal_pose.header.stamp = self.get_clock().now().to_msg()
            right_goal_pose.header.frame_id = "world"
            right_goal_pose.pose.position.x = float(
                self.grasing_starting_position["right"].pose.position.x
            )
            right_goal_pose.pose.position.y = float(
                self.grasing_starting_position["right"].pose.position.y
            )
            right_goal_pose.pose.position.z = float(
                self.ee_base_frame_right.pose.position.z - 0.15 * self.dt
            )
            right_goal_pose.pose.orientation.w = 1.0

            self.goal_pose_pub_left.publish(left_goal_pose)
            self.goal_pose_pub_right.publish(right_goal_pose)
        

    def rope_state_callback(self, msg):
        if self.gripper_closed:
            self.get_logger().info("Gripped closed, done")
            return

        if self.grasping_procedure and not self.gripper_closed:
            self.get_logger().info("Starting gripping procedure")
            self.grasp()
            return

        if self.num_iterations >= 300:
            self.get_logger().info("Done")
            return

        if self.ee_base_frame_right is None or self.ee_base_frame_left is None:
            self.get_logger().info("No ee base frame pose")
            return

        if len(msg.poses) == 0:
            self.get_logger().info("Skipping, no poses")
            return

        grip_positions = self.get_latest_grip_positions()

        if grip_positions is None:
            self.get_logger().warn("No left/right end effector poses yet. Skipping solve.")
            return

        callback_start = time.time()

        raw_points = np.asarray(
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

        sampled = resample_fixed_link_length_extend(
            raw_points,
            self.env.params.num_nodes,
            link_length=self.env.params.segment_length,
        )

        closest_dist_left, closest_seg_idx, closest_t, closest_point = (
            self.closest_point_on_rope_segments(sampled, grip_positions[0])
        )

        closest_dist_right, closest_seg_idx, closest_t, closest_point = (
            self.closest_point_on_rope_segments(sampled, grip_positions[1])
        )

        self.get_logger().info(
            f"Closest LEFT rope point: dist={closest_dist_left:.4f} m, grip={grip_positions[0]}"
        )
        self.get_logger().info(
            f"Closest RIGHT rope point: dist={closest_dist_right:.4f} m, grip={grip_positions[1]}"
        )

        if closest_dist_left <= 0.031 and closest_dist_right <= 0.031:
            self.grasping_procedure = True
            return

        if sampled is None:
            self.get_logger().warn("Could not resample rope_state.")
            return

        if self.state is None:
            self.x_grip = jnp.asarray(grip_positions)
            self.state = self.env.state(x_grip=self.x_grip)

            self.state = state_from_rope_points(
                self.env,
                self.state,
                sampled,
                grip_positions=grip_positions,
            )

            self.get_logger().info(
                "Initialized state from first rope state and real gripper pose."
            )
        else:
            x_node, x_weld, x_grip = self.env.unpack_state(self.state)

            grip_positions_jnp = jnp.asarray(grip_positions)

            self.state = jnp.concatenate(
                [
                    x_node.reshape(-1),
                    x_weld.reshape(-1),
                    grip_positions_jnp.reshape(-1),
                ]
            )

        if self.controller is None:
            self.initialize_controller()

        self.env.visualize(self.server, self.state)

        if self.first_solve:
            X_in = jnp.tile(self.state[None, :], (self.N + 1, 1))
            U_in = jnp.tile(self.control0[None, :], (self.N, 1))
            U_in = U_in.at[:, 2].set(-0.2)
            U_in = U_in.at[:, 5].set(-0.2)

            # GROUND_Z = 0.0

            # for i in range(U_in.shape[0]):
            #     x_curr = X_in[i]

            #     # unpack current grip position
            #     _, _, x_grip = self.env.unpack_state(x_curr)

            #     grip_z = x_grip[0, 2]

            #     u = U_in[i]

            #     # prevent commanding downward motion once at/below ground
            #     u = jax.lax.cond(
            #         grip_z <= GROUND_Z,
            #         lambda uu: uu.at[2].set(jnp.maximum(uu[2], 0.0)),
            #         lambda uu: uu,
            #         u,
            #     )

            #     X_in = X_in.at[i + 1].set(self.env.step(x_curr, u))
            #     U_in = U_in.at[i].set(u)

            self.controller.X0 = X_in
            self.controller.U0 = U_in

            self.first_solve = False

        if (
            self.latest_ee_pos_seq_left
            <= self.last_solve_latest_ee_pos_seq_left
            or self.latest_ee_pos_seq_right
            <= self.last_solve_latest_ee_pos_seq_right
            or self.ee_base_frame_seq_left
            <= self.last_solve_ee_base_frame_seq_left
            or self.ee_base_frame_seq_right
            <= self.last_solve_ee_base_frame_seq_right
        ):
            self.get_logger().info(
                "Skipping solve because one or more left/right EE poses "
                "or base-frame poses have not updated since last solve."
            )
            return

        self.last_solve_latest_ee_pos_seq_left = (
            self.latest_ee_pos_seq_left
        )

        self.last_solve_latest_ee_pos_seq_right = (
            self.latest_ee_pos_seq_right
        )

        self.last_solve_ee_base_frame_seq_left = (
            self.ee_base_frame_seq_left
        )

        self.last_solve_ee_base_frame_seq_right = (
            self.ee_base_frame_seq_right
        )

        solve_start = time.time()
        out = self.controller.run(
            x0=self.state,
            reference=None,
            parameter=None,
            Xi=jnp.zeros((self.state.shape[0], self.state.shape[0])),
        )

        solve_time = time.time() - solve_start
        U = out[2]
        # u0 = out[0]
        U.block_until_ready()

        u0_np = np.asarray(U[0])
        u1_np = np.asarray(U[1])

        if not np.isfinite(u0_np).all():
            self.get_logger().warn("u0 has NaN/Inf. Skipping.")
            return

        u0_np = np.clip(u0_np, -np.asarray(self.u_max), np.asarray(self.u_max))
        u1_np = np.clip(u1_np, -np.asarray(self.u_max), np.asarray(self.u_max))
        # control = u0_np * 1.0
        # control = u0_np + u1_np
        control = u0_np * 2.0

        left_control = control[0:3]
        right_control = control[3:6]
        # if np.linalg.norm(u0_np[:3]) <= 0.5:
        #     control += u1_np
        # control = np.clip(control, -np.asarray(self.u_max), np.asarray(self.u_max))
        callback_time = time.time() - callback_start


        left_goal_pose = PoseStamped()
        left_goal_pose.header.stamp = self.get_clock().now().to_msg()
        left_goal_pose.header.frame_id = "world"
        left_goal_pose.pose.position.x = float(
            self.ee_base_frame_left.pose.position.x + left_control[0] * self.dt
        )
        left_goal_pose.pose.position.y = float(
            self.ee_base_frame_left.pose.position.y + left_control[1] * self.dt
        )
        left_goal_pose.pose.position.z = float(
            self.ee_base_frame_left.pose.position.z + left_control[2] * self.dt
        )
        left_goal_pose.pose.orientation.w = 1.0

        right_goal_pose = PoseStamped()
        right_goal_pose.header.stamp = self.get_clock().now().to_msg()
        right_goal_pose.header.frame_id = "world"
        right_goal_pose.pose.position.x = float(
            self.ee_base_frame_right.pose.position.x + right_control[0] * self.dt
        )
        right_goal_pose.pose.position.y = float(
            self.ee_base_frame_right.pose.position.y + right_control[1] * self.dt
        )
        right_goal_pose.pose.position.z = float(
            self.ee_base_frame_right.pose.position.z + right_control[2] * self.dt
        )
        right_goal_pose.pose.orientation.w = 1.0

        self.goal_pose_pub_left.publish(left_goal_pose)
        self.goal_pose_pub_right.publish(right_goal_pose)

        self.get_logger().info("=" * 80)

        self.get_logger().info(
            f"LEFT goal pose: "
            f"x={self.ee_base_frame_left.pose.position.x + left_control[0] * self.dt:.4f} "
            f"y={self.ee_base_frame_left.pose.position.y + left_control[1] * self.dt:.4f} "
            f"z={self.ee_base_frame_left.pose.position.z + left_control[2] * self.dt:.4f}"
        )

        self.get_logger().info(
            f"RIGHT goal pose: "
            f"x={self.ee_base_frame_right.pose.position.x + right_control[0] * self.dt:.4f} "
            f"y={self.ee_base_frame_right.pose.position.y + right_control[1] * self.dt:.4f} "
            f"z={self.ee_base_frame_right.pose.position.z + right_control[2] * self.dt:.4f}"
        )

        self.get_logger().info(
            f"Received rope_state with {raw_points.shape[0]} raw points"
        )

        self.get_logger().info(
            f"Current LEFT EE pos: {grip_positions[0]}"
        )

        self.get_logger().info(
            f"Current RIGHT EE pos: {grip_positions[1]}"
        )

        self.get_logger().info(
            f"MPC solve time: {solve_time * 1000:.2f} ms"
        )

        self.get_logger().info(
            f"Total callback time: {callback_time * 1000:.2f} ms"
        )

        self.get_logger().info(f"u0_left: {u0_np[0:3]}")
        self.get_logger().info(f"u0_right:  {u0_np[4:6]}")

        self.get_logger().info(f"control_left:  {left_control}")
        self.get_logger().info(f"control_right: {right_control}")

        self.get_logger().info("=" * 80)
        # time.sleep(0.01)


def main(args=None):
    rclpy.init(args=args)

    node = RopeStateSolverNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()