#!/usr/bin/env python3

import sys
import termios
import tty
import select
import copy

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class IiwaPoseTeleopNode(Node):
    def __init__(self):
        super().__init__("left_iiwa_pose_teleop_node")

        self.sub = self.create_subscription(
            PoseStamped,
            "/left/end_effector_pose",
            self.pose_callback,
            10,
        )

        self.pub = self.create_publisher(
            PoseStamped,
            "/left/iiwa/goal_pose",
            10,
        )

        self.current_pose = None
        self.goal_pose = None

        self.step = 0.005  # 2 mm

        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info("Waiting for /end_effector_pose ...")
        self.print_help()

    def print_help(self):
        print("""
Keyboard teleop:
  w/s : +x / -x
  a/d : +y / -y
  q/e : +z / -z

  +/- : increase/decrease step
  space : publish current goal again
  ESC or Ctrl-C : quit

Current step: %.4f m
""" % self.step)

    def pose_callback(self, msg: PoseStamped):
        self.current_pose = msg

        if self.goal_pose is None:
            self.goal_pose = copy.deepcopy(msg)
            self.get_logger().info("Initialized goal pose from current pose.")

    def get_key(self):
        if select.select([sys.stdin], [], [], 0.0)[0]:
            return sys.stdin.read(1)
        return None

    def timer_callback(self):
        if self.goal_pose is None:
            return

        key = self.get_key()
        if key is None:
            return

        if key == "\x1b":  # ESC
            self.get_logger().info("Exiting teleop.")
            rclpy.shutdown()
            return

        moved = True

        if key == "w":
            self.goal_pose.pose.position.x += self.step
        elif key == "s":
            self.goal_pose.pose.position.x -= self.step
        elif key == "a":
            self.goal_pose.pose.position.y += self.step
        elif key == "d":
            self.goal_pose.pose.position.y -= self.step
        elif key == "q":
            self.goal_pose.pose.position.z += self.step
        elif key == "e":
            self.goal_pose.pose.position.z -= self.step
        elif key == " ":
            moved = True
        else:
            return

        if moved:
            self.publish_goal()

    def publish_goal(self):
        goal = copy.deepcopy(self.goal_pose)
        goal.header.stamp = self.get_clock().now().to_msg()

        if goal.header.frame_id == "":
            goal.header.frame_id = "world"

        self.pub.publish(goal)

        p = goal.pose.position
        self.get_logger().info(
            f"Published goal: x={p.x:.4f}, y={p.y:.4f}, z={p.z:.4f}"
        )


def main(args=None):
    rclpy.init(args=args)

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    node = IiwaPoseTeleopNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()