#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
import time

class MoveDown2mmNode(Node):
    def __init__(self):
        super().__init__("move_down_2mm_node")

        self.sub = self.create_subscription(
            PoseStamped,
            "/end_effector_pose",
            self.pose_callback,
            10,
        )

        self.pub = self.create_publisher(
            PoseStamped,
            "/iiwa/goal_pose",
            10,
        )

        self.move_count = 0
        self.max_moves = 10

        self.pose = None

        self.timer = self.create_timer(
            0.01,  # s
            self.timer_callback
        )

        self.get_logger().info(
            "Waiting for /end_effector_pose ..."
        )

    def timer_callback(self):
        if self.move_count >= self.max_moves:
            return
        if self.pose is None:
            return
        
        position = self.pose.position
        position.z -= 0.002

        goal = PoseStamped()

        goal.header.stamp = self.get_clock().now().to_msg()
        # goal.header.frame_id = msg.header.frame_id

        # Copy orientation
        # goal.pose.orientation = msg.pose.orientation

        # Move 2 mm downward from CURRENT pose
        # goal.pose.position.x = msg.pose.position.x
        # goal.pose.position.y = msg.pose.position.y
        # goal.pose.position.z = msg.pose.position.z + 0.01

        goal.pose = self.pose
        self.pub.publish(goal)

        self.move_count += 1

        self.get_logger().info(
            f"Move {self.move_count}/{self.max_moves}: "
            f"Published goal pose: "
            f"x={goal.pose.position.x:.4f}, "
            f"y={goal.pose.position.y:.4f}, "
            f"z={goal.pose.position.z:.4f}"
        )
        if self.move_count >= self.max_moves:
            self.get_logger().info(f"Completed {self.max_moves} moves. Stopping.")

    def pose_callback(self, msg: PoseStamped):
        if self.pose is not None:
            return
        self.pose = msg.pose


        


def main(args=None):
    rclpy.init(args=args)

    node = MoveDown2mmNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()