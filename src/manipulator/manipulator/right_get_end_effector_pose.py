#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped

from drake import lcmt_iiwa_status
from pydrake.all import (
    DrakeLcm,
    MultibodyPlant,
    Parser,
    RigidTransform,
    RollPitchYaw,
)


class IiwaFK:
    def __init__(
        self,
        model_url="package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf",
    ):
        self.plant = MultibodyPlant(time_step=0.0)

        parser = Parser(self.plant)

        models = parser.AddModelsFromUrl(model_url)
        assert len(models) == 1

        self.iiwa = models[0]

        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetFrameByName("iiwa_link_0", self.iiwa),
            RigidTransform(),
        )

        self.plant.Finalize()

        self.context = self.plant.CreateDefaultContext()

        self.link7 = self.plant.GetBodyByName(
            "iiwa_link_7",
            self.iiwa,
        )

        self.link7_frame = self.link7.body_frame()

    def fk_pose(self, q: np.ndarray):
        q = np.asarray(q, dtype=float).reshape(7)

        self.plant.SetPositions(
            self.context,
            self.iiwa,
            q,
        )

        X_W_EE = self.link7_frame.CalcPoseInWorld(
            self.context,
        )

        return X_W_EE


class EndEffectorPosePublisher(Node):

    def __init__(self):
        super().__init__("right_iiwa_end_effector_pose_publisher")

        self.declare_parameter("publish_rate", 500.0)

        publish_rate = float(
            self.get_parameter("publish_rate").value
        )

        self.status_channel = "IIWA_STATUS_2"

        self.get_logger().info(
            f"Listening to LCM channel: {self.status_channel}"
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            "/right/end_effector_pose",
            1,
        )


        self.work_station_pose_pub = self.create_publisher(
            PoseStamped,
            "/right/workstation/end_effector_pose",
            1,
        )

        self.fk = IiwaFK()

        self.lcm = DrakeLcm()

        self.latest_q = None

        self.lcm.Subscribe(
            self.status_channel,
            self.lcm_callback,
        )

        timer_period = 1.0 / publish_rate

        self.timer = self.create_timer(
            timer_period,
            self.timer_callback,
        )

    def lcm_callback(self, data):
        msg = lcmt_iiwa_status.decode(data)

        self.latest_q = np.asarray(
            msg.joint_position_measured,
            dtype=float,
        ).reshape(7)

    def timer_callback(self):

        #
        # Process pending LCM messages
        #
        self.lcm.HandleSubscriptions(1)
        # self.get_logger().info(f"{self.latest_q}")
        if self.latest_q is None:
            return

        X_W_EE = self.fk.fk_pose(self.latest_q)

        p = X_W_EE.translation()
        quat = X_W_EE.rotation().ToQuaternion()

        pose_msg = PoseStamped()

        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "world"

        pose_msg.pose.position.x = float(p[0])
        pose_msg.pose.position.y = float(p[1])
        pose_msg.pose.position.z = float(p[2])

        pose_msg.pose.orientation.w = float(quat.w())
        pose_msg.pose.orientation.x = float(quat.x())
        pose_msg.pose.orientation.y = float(quat.y())
        pose_msg.pose.orientation.z = float(quat.z())

        self.pose_pub.publish(pose_msg)

        # rpy_deg = np.rad2deg(
        #     RollPitchYaw(X_W_EE.rotation()).vector()
        # )


        # TODO: Fix these
        workstation_pose_msg = PoseStamped()
        workstation_pose_msg.header.stamp = self.get_clock().now().to_msg()
        workstation_pose_msg.header.frame_id = "workstation"
        workstation_pose_msg.pose.position.x = float(p[0]) - 0.405
        workstation_pose_msg.pose.position.y = float(p[1]) - 0.15 - 0.5
        workstation_pose_msg.pose.position.z = float(p[2]) - 0.22
        workstation_pose_msg.pose.orientation.w = float(quat.w())
        workstation_pose_msg.pose.orientation.x = float(quat.x())
        workstation_pose_msg.pose.orientation.y = float(quat.y())
        workstation_pose_msg.pose.orientation.z = float(quat.z())
        self.work_station_pose_pub.publish(workstation_pose_msg)
        

        # self.get_logger().info(
        #     f"p = {np.round(p, 4)} | "
        #     f"rpy_deg = {np.round(rpy_deg, 2)}",
        #     throttle_duration_sec=1.0,
        # )


def main():
    rclpy.init()

    node = EndEffectorPosePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()