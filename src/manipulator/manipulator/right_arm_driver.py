#!/usr/bin/env python3

import time
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState

from drake import lcmt_iiwa_status, lcmt_iiwa_command
from pydrake.all import DrakeLcm


class SafeIiwaJointExecutor(Node):
    def __init__(self):
        super().__init__("right_safe_iiwa_joint_executor")

        self.declare_parameter("lcm_status_channel", "IIWA_STATUS_2")
        self.declare_parameter("lcm_command_channel", "IIWA_COMMAND_2")
        self.declare_parameter("max_joint_delta_deg", 3.0)
        self.declare_parameter("status_timeout_sec", 0.2)

        self.input_topic = "/right/iiwa/joint_position_command"
        self.lcm_status_channel = self.get_parameter("lcm_status_channel").value
        self.lcm_command_channel = self.get_parameter("lcm_command_channel").value
        self.max_joint_delta_deg = float(
            self.get_parameter("max_joint_delta_deg").value
        )
        self.status_timeout_sec = float(
            self.get_parameter("status_timeout_sec").value
        )

        self.lcm = DrakeLcm()

        self.sub = self.create_subscription(
            JointState,
            self.input_topic,
            self.command_callback,
            1,
        )

        self.get_logger().info(f"Listening for joint commands on {self.input_topic}")
        self.get_logger().info(f"Reading state from LCM {self.lcm_status_channel}")
        self.get_logger().info(f"Publishing commands to LCM {self.lcm_command_channel}")
        self.get_logger().info(
            f"Rejecting commands with joint delta > {self.max_joint_delta_deg:.2f} deg"
        )

    def read_current_position(self):
        q_out = {"q": None}

        def handler(data):
            msg = lcmt_iiwa_status.decode(data)
            q_out["q"] = np.asarray(
                msg.joint_position_measured,
                dtype=float,
            ).reshape(7)

        self.lcm.Subscribe(self.lcm_status_channel, handler)

        t0 = time.time()
        while time.time() - t0 < self.status_timeout_sec:
            self.lcm.HandleSubscriptions(10)
            if q_out["q"] is not None:
                return q_out["q"]

        raise RuntimeError(
            f"Failed to receive {self.lcm_status_channel} "
            f"within {self.status_timeout_sec} sec"
        )

    def publish_lcm_position_command(self, q_cmd):
        # return
        msg = lcmt_iiwa_command()
        msg.utime = int(time.time() * 1e6)
        msg.num_joints = 7
        msg.joint_position = np.asarray(q_cmd, dtype=float).reshape(7).tolist()
        msg.joint_torque = np.zeros(7).tolist()

        self.lcm.Publish(self.lcm_command_channel, msg.encode())

    def hold_current_position(self, q_now):
        self.publish_lcm_position_command(q_now)
        self.get_logger().warn("STOPPED: holding current measured joint position.")

    def command_callback(self, msg: JointState):
        if len(msg.position) != 7:
            self.get_logger().warn(
                f"Rejected command: expected 7 joints, got {len(msg.position)}"
            )
            return

        q_cmd = np.asarray(msg.position, dtype=float).reshape(7)

        try:
            q_now = self.read_current_position()
        except RuntimeError as e:
            self.get_logger().warn(str(e))
            return

        joint_delta_deg = np.rad2deg(np.abs(q_cmd - q_now))
        max_delta_deg = float(np.max(joint_delta_deg))
        worst_joint = int(np.argmax(joint_delta_deg)) + 1

        if max_delta_deg > self.max_joint_delta_deg:
            self.get_logger().error(
                f"Rejected unsafe joint command. "
                f"Joint {worst_joint} delta is {max_delta_deg:.3f} deg, "
                f"limit is {self.max_joint_delta_deg:.3f} deg. "
                f"All deltas deg: {np.round(joint_delta_deg, 3)}"
            )
            self.hold_current_position(q_now)
            return

        self.publish_lcm_position_command(q_cmd)

        self.get_logger().info(
            f"Executed joint command. "
            f"Max joint delta: {max_delta_deg:.3f} deg"
        )


def main(args=None):
    rclpy.init(args=args)

    node = SafeIiwaJointExecutor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()