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
        self.declare_parameter(
            "end_effector_pose_topic",
            "/right/workstation/end_effector_pose",
        )

        self.rope_state_topic = str(self.get_parameter("rope_state_topic").value)
        self.end_effector_pose_topic = str(
            self.get_parameter("end_effector_pose_topic").value
        )

        self.latest_ee_pos = None

        self.latest_ee_pos_seq = 0
        self.ee_base_frame_seq = 0

        self.last_solve_latest_ee_pos_seq = -1
        self.last_solve_ee_base_frame_seq = -1

        self.server = viser.ViserServer()
        _ = self.server.scene.add_grid(name="ground")

        self.env = RopeEnv(
            time_step=0.02,
            num_segments=10,
            rope_length=1.0,
            rope_diameter=0.01,
            youngs_modulus=1e5,
            mass_density=300,
            num_floating_grippers=1,
            grip_stiffness=100,
            gripper_radius=0.02,
            contact_smoothing=1e-3,
        )

        self.N = 50
        self.dt = self.env.params.dt

        self.x_grip = None
        self.state = None
        self.first_solve = True
        self.controller = None

        self.control0 = self.env.control(
            c_grip=jnp.array([1.0]),
        )

        # x_coords = (
        #     jnp.arange(self.env.params.num_nodes)
        #     * self.env.params.segment_length
        #     - 0.2
        # )

        # goal_nodes = jnp.stack(
        #     (
        #         x_coords,
        #         jnp.zeros_like(x_coords),
        #         jnp.zeros_like(x_coords),
        #     ),
        #     axis=1,
        # )
        y_coords = (
            jnp.arange(self.env.params.num_nodes) * self.env.params.segment_length
            - 0.5
        )

        # Make the right side higher and gradually slope downward
        z_coords = jnp.linspace(
            0.0,   # left end height
            0.3,   # right end height
            self.env.params.num_nodes,
        )

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
        self.u_max = jnp.array([vmax, vmax, vmax, 10.0])

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

        self.goal_pose_pub = self.create_publisher(
            PoseStamped,
            '/right/iiwa/goal_pose',
            1,
        )

        self.ee_sub = self.create_subscription(
            PoseStamped,
            self.end_effector_pose_topic,
            self.end_effector_pose_callback,
            1,
        )

        self.ee_base_frame_sub = self.create_subscription(
            PoseStamped,
            '/right/end_effector_pose',
            self.ee_base_frame_callback,
            1
        )

        self.grip_pub = self.create_publisher(
            Bool,
            '/right_grip',
            1
        )

        self.get_logger().info(f"Subscribed to {self.rope_state_topic}")
        self.get_logger().info(f"Subscribed to {self.end_effector_pose_topic}")
        self.get_logger().info("Waiting for first real gripper pose before initializing state.")

        self.num_iterations = 0
        self.ee_base_frame = None
        self.grasping_procedure = False
        self.gripper_closed = False
        self.grasing_starting_position = None

    def ee_base_frame_callback(self, msg):
        self.ee_base_frame = msg
        self.ee_base_frame_seq += 1

    def end_effector_pose_callback(self, msg):
        p = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float32,
        )

        self.latest_ee_pos = p
        self.latest_ee_pos_seq += 1

    def get_latest_ee_pos(self):
        if self.latest_ee_pos is None:
            return None
        return self.latest_ee_pos.copy()

    def initialize_controller(self):
        def cost(W, reference, x, u, t):
            state_err = x - self.state_goal
            control_err = u - self.control0

            return (
                1.0 * jnp.sum(state_err[:-3] ** 2)
                + 0.1 * jnp.sum(control_err[:-1] ** 2)
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
        # This will all work in the workstation frame
        ee_pos = self.get_latest_ee_pos()
        if self.grasing_starting_position is None:
            self.grasing_starting_position = self.ee_base_frame
        
        if ee_pos[2] <= 0.005 and not self.gripper_closed:
            self.grip_pub.publish(Bool(data=True))
            self.gripper_closed = True
            self.get_logger().info("Gripping")
            time.sleep(3)
            self.get_logger().info("Finish Gripping")
            return
        
        if not self.gripper_closed and ee_pos[2] > 0.005:
            goal_pose = PoseStamped()
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            goal_pose.header.frame_id = "world"
            goal_pose.pose.position.x = float(self.grasing_starting_position.pose.position.x)
            goal_pose.pose.position.y = float(self.grasing_starting_position.pose.position.y)
            goal_pose.pose.position.z = float(self.ee_base_frame.pose.position.z - 0.15 * self.dt)
            goal_pose.pose.orientation.w = 1.0

            self.goal_pose_pub.publish(goal_pose)
            return
        

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

        if self.ee_base_frame is None:
            self.get_logger().info("No ee base frame pose")
            return

        if len(msg.poses) == 0:
            self.get_logger().info("Skipping, no poses")
            return

        ee_pos = self.get_latest_ee_pos()

        if ee_pos is None:
            self.get_logger().warn("No end effector pose yet. Skipping solve.")
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

        closest_dist, closest_seg_idx, closest_t, closest_point = (
            self.closest_point_on_rope_segments(sampled, ee_pos)
        )

        self.get_logger().info(
            f"Closest rope point to gripper: "
            f"dist={closest_dist:.4f} m, "
            f"segment={closest_seg_idx}->{closest_seg_idx + 1}, "
            f"t={closest_t:.3f}, "
            f"point={closest_point}, "
            f"grip={ee_pos}"
        )

        if closest_dist <= 0.031:
            self.grasping_procedure = True
            return

        if sampled is None:
            self.get_logger().warn("Could not resample rope_state.")
            return

        if self.state is None:
            ee_grip = jnp.asarray(ee_pos)

            if ee_grip.shape == (3,):
                ee_grip = ee_grip[None, :]

            self.x_grip = ee_grip
            self.state = self.env.state(x_grip=self.x_grip)

            self.state = state_from_rope_points(
                self.env,
                self.state,
                sampled,
                grip_position=ee_pos,
            )

            self.get_logger().info(
                "Initialized state from first rope state and real gripper pose."
            )
        else:
            x_node, x_weld, x_grip = self.env.unpack_state(self.state)

            ee_grip = jnp.asarray(ee_pos)
            if ee_grip.shape == (3,):
                ee_grip = ee_grip[None, :]

            self.state = jnp.concatenate(
                [
                    x_node.reshape(-1),      # keep first rope constant
                    x_weld.reshape(-1),
                    ee_grip.reshape(-1),     # update manipulator/gripper state
                ]
            )

        if self.controller is None:
            self.initialize_controller()

        self.env.visualize(self.server, self.state)

        if self.first_solve:
            X_in = jnp.tile(self.state[None, :], (self.N + 1, 1))
            U_in = jnp.tile(self.control0[None, :], (self.N, 1))
            U_in = U_in.at[:30, 2].set(-0.2)

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
            self.latest_ee_pos_seq <= self.last_solve_latest_ee_pos_seq
            or self.ee_base_frame_seq <= self.last_solve_ee_base_frame_seq
        ):
            self.get_logger().info(
                "Skipping solve because latest_ee_pos and/or ee_base_frame "
                "has not updated since the last full solve."
            )
            return

        self.last_solve_latest_ee_pos_seq = self.latest_ee_pos_seq
        self.last_solve_ee_base_frame_seq = self.ee_base_frame_seq

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
        control = u0_np * 2
        # if np.linalg.norm(u0_np[:3]) <= 0.5:
        #     control += u1_np
        # control = np.clip(control, -np.asarray(self.u_max), np.asarray(self.u_max))
        callback_time = time.time() - callback_start

        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = "world"
        goal_pose.pose.position.x = float(self.ee_base_frame.pose.position.x + control[0] * self.dt)
        goal_pose.pose.position.y = float(self.ee_base_frame.pose.position.y + control[1] * self.dt)
        goal_pose.pose.position.z = float(self.ee_base_frame.pose.position.z + control[2] * self.dt)
        goal_pose.pose.orientation.w = 1.0

        self.num_iterations += 1
        self.goal_pose_pub.publish(goal_pose)

        self.get_logger().info("=" * 80)
        self.get_logger().info(
            f"Publishing goal pose "
            f"x={ee_pos[0] + control[0] * self.dt} "
            f"y={ee_pos[1] + control[1] * self.dt} "
            f"z={ee_pos[2] + control[2] * self.dt}"
        )
        self.get_logger().info(
            f"Received rope_state with {raw_points.shape[0]} raw points"
        )
        self.get_logger().info(f"Current EE pos: {ee_pos}")
        self.get_logger().info(
            f"MPC solve time: {solve_time * 1000:.2f} ms"
        )
        self.get_logger().info(
            f"Total callback time: {callback_time * 1000:.2f} ms"
        )
        self.get_logger().info(f"u0: {u0_np}")
        self.get_logger().info(f"control: {control}")
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