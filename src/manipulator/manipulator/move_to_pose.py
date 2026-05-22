#!/usr/bin/env python3

import time
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from drake import lcmt_iiwa_status
from pydrake.all import DrakeLcm

from manipulator.iiwa_cartesian_ik import SingleIiwaPositionIK


class IiwaMpcGoalPoseNode(Node):
    def __init__(self):
        super().__init__("iiwa_mpc_goal_pose_node")

        self.declare_parameter("goal_topic", "/iiwa/goal_pose")
        self.declare_parameter("command_topic", "/iiwa/joint_position_command")
        self.declare_parameter("lcm_status_channel", "IIWA_STATUS")

        self.declare_parameter("position_tol", 0.001)
        self.declare_parameter("max_joint_step_deg", 12.0)
        self.declare_parameter("max_cartesian_step_m", 0.1)

        self.goal_topic = self.get_parameter("goal_topic").value
        self.command_topic = self.get_parameter("command_topic").value
        self.lcm_status_channel = self.get_parameter("lcm_status_channel").value

        self.position_tol = float(self.get_parameter("position_tol").value)
        self.max_joint_step_deg = float(self.get_parameter("max_joint_step_deg").value)
        self.max_cartesian_step_m = float(
            self.get_parameter("max_cartesian_step_m").value
        )

        self.ik = SingleIiwaPositionIK()

        self.joint_names = [
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7",
        ]

        self.goal_sub = self.create_subscription(
            PoseStamped,
            self.goal_topic,
            self.goal_callback,
            10,
        )

        self.command_pub = self.create_publisher(
            JointState,
            self.command_topic,
            10,
        )

        self.get_logger().info(f"Listening on {self.goal_topic}")
        self.get_logger().info(f"Publishing to {self.command_topic}")
        self.get_logger().info(
            f"Rejecting Cartesian steps > {self.max_cartesian_step_m * 1000.0:.2f} mm"
        )

    def read_current_iiwa_position(self, timeout_sec=0.2):
        lcm = DrakeLcm()
        q_out = {"q": None}

        def handler(data):
            msg = lcmt_iiwa_status.decode(data)
            q_out["q"] = np.asarray(
                msg.joint_position_measured,
                dtype=float,
            ).reshape(7)

        lcm.Subscribe(self.lcm_status_channel, handler)

        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            lcm.HandleSubscriptions(10)
            if q_out["q"] is not None:
                return q_out["q"]


        raise RuntimeError(
            f"Failed to receive {self.lcm_status_channel} within {timeout_sec} sec"
        )

    def goal_callback(self, msg: PoseStamped):
        p_goal = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=float,
        )
        try:
            q_now = self.read_current_iiwa_position()
        except RuntimeError as e:
            self.get_logger().warn(str(e))
            return
        p_now = self.ik.fk_position(q_now)
        cartesian_step = float(np.linalg.norm(p_goal - p_now))

        if cartesian_step > self.max_cartesian_step_m + 0.001:
            self.get_logger().warn(
                f"Rejected goal pose: step is {cartesian_step * 1000.0:.2f} mm, "
                f"limit is {self.max_cartesian_step_m * 1000.0:.2f} mm"
            )
            return

        q_cmd, info = self.ik.solve_position_ik(
            p_WQ=p_goal,
            q_seed=q_now,
            position_tol=self.position_tol,
            max_joint_move_from_seed=np.deg2rad(self.max_joint_step_deg),
            min_sigma=1e-4,
            max_cond=1e4,
        )

        if q_cmd is None:
            self.get_logger().warn(f"IK failed: {info}")
            return
        

        max_joint_step_deg = float(np.rad2deg(np.max(np.abs(q_cmd - q_now))))

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = self.joint_names
        out.position = q_cmd.tolist()

        self.command_pub.publish(out)

        self.get_logger().info(
            f"Published q_cmd. "
            f"Cartesian step: {cartesian_step * 1000.0:.2f} mm, "
            f"max joint step: {max_joint_step_deg:.3f} deg, "
            f"IK error: {info['pos_err'] * 1000.0:.3f} mm"
            f"{q_cmd}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = IiwaMpcGoalPoseNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()