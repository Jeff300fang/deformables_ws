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

from environments import ClothEnv
from planners import gpu_sls

def make_control_constraints(
    u_min: jnp.ndarray,
    u_max: jnp.ndarray,
):
    def constraints(x, u, t):
        control_constraints = jnp.concatenate((u - u_max, u_min - u))
        return control_constraints

    return constraints

def make_constant_disturbance(alpha):
    def disturbance(X):
        N, nx = X.shape
        E0 = alpha * jnp.eye(nx, dtype=X.dtype)
        return jnp.broadcast_to(E0, (N, nx, nx))

    return disturbance

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


class ClothPlanner(Node):
    def __init__(self):
        super().__init__("cloth_planner")

        self.cloth_sub = self.create_subscription(
            PoseArray,
             "/front/sam_cloth/cloth_grid_poses",
             self.cloth_callback,
             1
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

        self.ee_base_frame_sub_right = self.create_subscription(
            PoseStamped,
            "/right/end_effector_pose",
            self.ee_base_frame_right_callback,
            1,
        )

        self.state = None

        self.env = ClothEnv.from_regular_grid(
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


        self.latest_ee_pos_left = None
        self.latest_ee_pos_right = None

        self.latest_ee_pos_seq_left = 0
        self.latest_ee_pos_seq_right = 0

        self.ee_base_frame_left = None
        self.ee_base_frame_right = None

        self.ee_base_frame_seq_left = 0
        self.ee_base_frame_seq_right = 0

        self.last_solve_latest_ee_pos_seq_left = -1
        self.last_solve_latest_ee_pos_seq_right = -1

        self.last_solve_ee_base_frame_seq_left = -1
        self.last_solve_ee_base_frame_seq_right = -1

        self.solve_save_dir = Path("cloth_solve_debug")
        self.solve_save_dir.mkdir(parents=True, exist_ok=True)

        self.N = 10
        self.dt = self.env.params.dt

        self.visualize_rollouts_enabled = True

        self.state_goal = None
        self.control0 = self.env.control(
            c_grip=jnp.array([1.0, 1.0]),
        )
        vmax = 0.2
        u_max = jnp.array([vmax, vmax, vmax, 10.0])
        self.u_max = jnp.repeat(u_max, 2)

        self.constraints = make_control_constraints(
            u_min=-self.u_max,
            u_max=self.u_max,
        )

        self.controller = None
        self.first_solve = True

        self.server = viser.ViserServer()
        self.server.scene.add_grid(name="/world_grid", position=(0.0, 0.0, 0.0))

        self.rollout_handles = []
        self.rollout_stride = 3
        self.rollout_point_size = 0.006
        self.rollout_line_width = 2.0
        self.num_iterations = 0
        self.prev_sampled = None

    def initialize_controller(self):
        def cost(W, reference, x, u, t):
            state_err = x - reference[:x.shape[-1]]
            control_err = u - reference[x.shape[-1]:]

            # state weighting
            q = jnp.ones_like(state_err[:-6])

            # rope node coordinates are interleaved [x, y, z]
            q = q.at[0::3].set(1.0)   # x weight
            q = q.at[1::3].set(1.0)    # y weight
            q = q.at[2::3].set(1.0)    # z weight

            state_cost = jnp.sum(q * (state_err[:-6] ** 2))

            control_cost = 0.1 * jnp.sum(control_err[:-2] ** 2)

            return state_cost + control_cost

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
            enable_fastsls=True,
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
            alpha=0.03 * self.dt,
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

    def clear_rollout_visualization(self):
        for h in self.rollout_handles:
            try:
                h.remove()
            except Exception:
                pass
        self.rollout_handles = []

    def make_cloth_edges(self, num_nodes: int):
        side = int(round(np.sqrt(num_nodes)))

        if side * side != num_nodes:
            return np.zeros((0, 2), dtype=np.int32)

        edges = []

        for r in range(side):
            for c in range(side):
                idx = r * side + c

                if c + 1 < side:
                    edges.append([idx, idx + 1])

                if r + 1 < side:
                    edges.append([idx, idx + side])

        return np.asarray(edges, dtype=np.int32)

    def visualize_rollouts(self, X, stride=1):
        if not self.visualize_rollouts_enabled:
            return

        if X is None:
            return

        X_np = np.asarray(X)

        if X_np.ndim != 2:
            self.get_logger().warn(f"Expected X shape (T, nx), got {X_np.shape}")
            return

        self.clear_rollout_visualization()

        T = X_np.shape[0]

        # Standard matplotlib-like tab colors, hardcoded to avoid dependency.
        tab_colors = np.asarray(
            [
                [31, 119, 180],
                [255, 127, 14],
                [44, 160, 44],
                [214, 39, 40],
                [148, 103, 189],
                [140, 86, 75],
                [227, 119, 194],
                [127, 127, 127],
                [188, 189, 34],
                [23, 190, 207],
            ],
            dtype=np.uint8,
        )

        for k in range(0, T, max(1, stride)):
            xk = jnp.asarray(X_np[k])
            x_node, x_grip = self.env.unpack_state(xk)

            nodes = np.asarray(x_node, dtype=np.float32)
            grips = np.asarray(x_grip, dtype=np.float32)

            if nodes.ndim != 2 or nodes.shape[1] != 3:
                self.get_logger().warn(f"Bad node shape at rollout step {k}: {nodes.shape}")
                continue

            color = tab_colors[k % len(tab_colors)]

            # Earlier states are more transparent-looking by blending toward white.
            alpha = 0.25 + 0.75 * (k / max(1, T - 1))
            blended = (
                alpha * color.astype(np.float32)
                + (1.0 - alpha) * np.array([255.0, 255.0, 255.0])
            ).astype(np.uint8)

            node_colors = np.tile(blended[None, :], (nodes.shape[0], 1))

            h_pts = self.server.scene.add_point_cloud(
                name=f"/rollout/step_{k:03d}/nodes",
                points=nodes,
                colors=node_colors,
                point_size=self.rollout_point_size,
            )
            self.rollout_handles.append(h_pts)

            edges = self.make_cloth_edges(nodes.shape[0])

            if edges.shape[0] > 0:
                segments = np.stack(
                    [
                        nodes[edges[:, 0]],
                        nodes[edges[:, 1]],
                    ],
                    axis=1,
                )

                seg_colors = np.tile(blended[None, :], (segments.shape[0], 1))

                h_edges = self.server.scene.add_line_segments(
                    name=f"/rollout/step_{k:03d}/edges",
                    points=segments,
                    colors=seg_colors,
                    line_width=self.rollout_line_width,
                )
                self.rollout_handles.append(h_edges)

            if grips.ndim == 2 and grips.shape[1] == 3:
                grip_colors = np.tile(
                    np.array([[0, 0, 0]], dtype=np.uint8),
                    (grips.shape[0], 1),
                )

                h_grips = self.server.scene.add_point_cloud(
                    name=f"/rollout/step_{k:03d}/grippers",
                    points=grips,
                    colors=grip_colors,
                    point_size=0.012,
                )
                self.rollout_handles.append(h_grips)

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

    def cloth_callback(self, msg):
        if self.num_iterations >= 1000:
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

        sampled = np.asarray(
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

        if sampled.shape[0] == 0:
            self.get_logger().warn("Missing points")
            return

        if self.state is None:
            self.x_grip = jnp.asarray(grip_positions)
            self.state = self.env.state(x_grip=self.x_grip)
            self.state = state_from_cloth_points(
                self.env,
                self.state,
                sampled_points=sampled,
                grip_positions=grip_positions
            )

        if self.controller is None:
            self.initialize_controller()

        if self.state_goal is None:
            nodes_goal = jnp.asarray(sampled)
            nodes_goal = nodes_goal.at[:, 0].set(jnp.abs(nodes_goal[:, 0]))
            nodes_goal = nodes_goal.at[:, 2].add(0.02)

            self.state_goal = state_from_cloth_points(
                self.env,
                self.state,
                sampled_points=nodes_goal,
                grip_positions=grip_positions,
            )
                
        # TODO: Maybe need to project grippers onto cloth here
        alpha = 0.7
        if self.prev_sampled is not None:
            sampled = alpha * self.prev_sampled + (1.0 - alpha) * sampled
        self.prev_sampled = sampled.copy()

        sampled_jnp = jnp.asarray(sampled)
        
        self.state = state_from_cloth_points(
            self.env,
            self.state,
            sampled_points=sampled_jnp,
            grip_positions=grip_positions
        )

        if self.first_solve:
            self.controller.X0 = jnp.tile(self.state[None, :], (self.N + 1, 1))
            self.controller.U0 = jnp.tile(self.control0[None, :], (self.N, 1))
            self.reference = jnp.concatenate([self.state_goal, self.control0], axis=0)
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
        self.controller.U0 = self.controller.U0.at[:, -2:].set(1.0)
        solve_start = time.time()
        out = self.controller.run(
            x0=self.state,
            reference=self.reference,
            parameter=None,
            Xi=jnp.zeros((self.state.shape[0], self.state.shape[0])),
        )

        solve_time = time.time() - solve_start

        X_sol = out[1]
        U_sol = out[2]

        save_path = self.solve_save_dir / f"solve_{self.num_iterations:05d}.npz"
        self.num_iterations += 1
        np.savez_compressed(
            save_path,
            iteration=np.array(self.num_iterations),
            state=np.asarray(self.state),
            X=np.asarray(X_sol),
            U=np.asarray(U_sol),
            solve_time=np.array(solve_time),
            left_ee_pos=np.asarray(self.latest_ee_pos_left),
            right_ee_pos=np.asarray(self.latest_ee_pos_right),
        )
        self.get_logger().info(f"Saved solve debug data to {save_path}")
        U_sol.block_until_ready()
        U = U_sol
        X = X_sol

        self.visualize_rollouts(X, stride=3)
        u0_np = np.asarray(U[0])
        u0_np = np.clip(u0_np, -np.asarray(self.u_max), np.asarray(self.u_max))
        control = u0_np
        left_control = control[0:3]
        right_control = control[3:6]
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
        self.goal_pose_pub_left.publish(left_goal_pose)
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
        self.goal_pose_pub_right.publish(right_goal_pose)

        callback_time = time.time() - callback_start

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
            f"Received cloth state with {sampled.shape[0]} raw points"
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
        self.get_logger().info(f"u0_right:  {u0_np[3:6]}")

        self.get_logger().info("=" * 80)
        time.sleep(0.2)

def main(args=None):
    rclpy.init(args=args)

    node = ClothPlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
