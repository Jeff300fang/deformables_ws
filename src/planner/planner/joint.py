#!/usr/bin/env python3

import sys
import termios
import tty
import select
import copy

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool


class DualIiwaPoseTeleopNode(Node):
    def __init__(self):
        super().__init__("dual_iiwa_pose_teleop_node")

        # =========================
        # Subscribers
        # =========================
        self.left_sub = self.create_subscription(
            PoseStamped,
            "/left/end_effector_pose",
            self.left_pose_callback,
            10,
        )

        self.right_sub = self.create_subscription(
            PoseStamped,
            "/right/end_effector_pose",
            self.right_pose_callback,
            10,
        )

        # =========================
        # Pose publishers
        # =========================
        self.left_pose_pub = self.create_publisher(
            PoseStamped,
            "/left/iiwa/goal_pose",
            10,
        )

        self.right_pose_pub = self.create_publisher(
            PoseStamped,
            "/right/iiwa/goal_pose",
            10,
        )

        # =========================
        # Grip publishers
        # =========================
        self.left_grip_pub = self.create_publisher(
            Bool,
            "/left_grip",
            10,
        )

        self.right_grip_pub = self.create_publisher(
            Bool,
            "/right_grip",
            10,
        )

        self.left_current_pose = None
        self.right_current_pose = None

        self.left_goal_pose = None
        self.right_goal_pose = None

        self.step = 0.005  # meters

        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info("Waiting for /left/end_effector_pose ...")
        self.get_logger().info("Waiting for /right/end_effector_pose ...")
        self.print_help()

    def print_help(self):
        print(
            """
Keyboard teleop:

LEFT MANIPULATOR:
  w/s : +x / -x
  a/d : +y / -y
  q/e : +z / -z

RIGHT MANIPULATOR:
  i/k : +x / -x
  j/l : +y / -y
  u/o : +z / -z

GRIPPERS:
  g : left grip
  r : left ungrip

  h : right grip
  y : right ungrip

OTHER:
  space : publish both current goals again
  ESC or Ctrl-C : quit

Current step: %.4f m
"""
            % self.step
        )

    # ============================================================
    # Pose callbacks
    # ============================================================

    def left_pose_callback(self, msg: PoseStamped):
        self.left_current_pose = msg

        if self.left_goal_pose is None:
            self.left_goal_pose = copy.deepcopy(msg)
            self.get_logger().info(
                "Initialized LEFT goal pose from current pose."
            )

    def right_pose_callback(self, msg: PoseStamped):
        self.right_current_pose = msg

        if self.right_goal_pose is None:
            self.right_goal_pose = copy.deepcopy(msg)
            self.get_logger().info(
                "Initialized RIGHT goal pose from current pose."
            )

    # ============================================================
    # Keyboard
    # ============================================================

    def get_key(self):
        if select.select([sys.stdin], [], [], 0.0)[0]:
            return sys.stdin.read(1)
        return None

    def timer_callback(self):
        key = self.get_key()
        if key is None:
            return

        if key == "\x1b":  # ESC
            self.get_logger().info("Exiting teleop.")
            rclpy.shutdown()
            return

        left_moved = False
        right_moved = False

        # ========================================================
        # Left manipulator: WASDQE
        # ========================================================

        if key in ["w", "s", "a", "d", "q", "e"]:
            if self.left_goal_pose is None:
                self.get_logger().warn("LEFT goal pose not initialized yet.")
                return

            if key == "w":
                self.left_goal_pose.pose.position.x += self.step
            elif key == "s":
                self.left_goal_pose.pose.position.x -= self.step
            elif key == "a":
                self.left_goal_pose.pose.position.y += self.step
            elif key == "d":
                self.left_goal_pose.pose.position.y -= self.step
            elif key == "q":
                self.left_goal_pose.pose.position.z += self.step
            elif key == "e":
                self.left_goal_pose.pose.position.z -= self.step

            left_moved = True

        # ========================================================
        # Right manipulator: IJKLUO
        # ========================================================

        elif key in ["i", "k", "j", "l", "u", "o"]:
            if self.right_goal_pose is None:
                self.get_logger().warn("RIGHT goal pose not initialized yet.")
                return

            if key == "i":
                self.right_goal_pose.pose.position.x += self.step
            elif key == "k":
                self.right_goal_pose.pose.position.x -= self.step
            elif key == "j":
                self.right_goal_pose.pose.position.y += self.step
            elif key == "l":
                self.right_goal_pose.pose.position.y -= self.step
            elif key == "u":
                self.right_goal_pose.pose.position.z += self.step
            elif key == "o":
                self.right_goal_pose.pose.position.z -= self.step

            right_moved = True

        # ========================================================
        # Gripper controls
        # ========================================================

        elif key == "g":
            self.publish_left_grip(True)

        elif key == "r":
            self.publish_left_grip(False)

        elif key == "h":
            self.publish_right_grip(True)

        elif key == "y":
            self.publish_right_grip(False)

        # ========================================================
        # Republish both
        # ========================================================

        elif key == " ":
            if self.left_goal_pose is not None:
                left_moved = True
            if self.right_goal_pose is not None:
                right_moved = True

        else:
            return

        if left_moved:
            self.publish_left_goal()

        if right_moved:
            self.publish_right_goal()

    # ============================================================
    # Goal publishers
    # ============================================================

    def publish_left_goal(self):
        if self.left_goal_pose is None:
            self.get_logger().warn("Cannot publish LEFT goal; not initialized.")
            return

        goal = copy.deepcopy(self.left_goal_pose)
        goal.header.stamp = self.get_clock().now().to_msg()

        if goal.header.frame_id == "":
            goal.header.frame_id = "world"

        self.left_pose_pub.publish(goal)

        p = goal.pose.position

        self.get_logger().info(
            f"Published LEFT goal: "
            f"x={p.x:.4f}, "
            f"y={p.y:.4f}, "
            f"z={p.z:.4f}"
        )

    def publish_right_goal(self):
        if self.right_goal_pose is None:
            self.get_logger().warn("Cannot publish RIGHT goal; not initialized.")
            return

        goal = copy.deepcopy(self.right_goal_pose)
        goal.header.stamp = self.get_clock().now().to_msg()

        if goal.header.frame_id == "":
            goal.header.frame_id = "world"

        self.right_pose_pub.publish(goal)

        p = goal.pose.position

        self.get_logger().info(
            f"Published RIGHT goal: "
            f"x={p.x:.4f}, "
            f"y={p.y:.4f}, "
            f"z={p.z:.4f}"
        )

    # ============================================================
    # Gripper publishers
    # ============================================================

    def publish_left_grip(self, grip: bool):
        msg = Bool()
        msg.data = grip

        self.left_grip_pub.publish(msg)

        if grip:
            self.get_logger().info("Published LEFT GRIP command.")
        else:
            self.get_logger().info("Published LEFT UNGRIP command.")

    def publish_right_grip(self, grip: bool):
        msg = Bool()
        msg.data = grip

        self.right_grip_pub.publish(msg)

        if grip:
            self.get_logger().info("Published RIGHT GRIP command.")
        else:
            self.get_logger().info("Published RIGHT UNGRIP command.")


def main(args=None):
    rclpy.init(args=args)

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    node = DualIiwaPoseTeleopNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        termios.tcsetattr(
            sys.stdin,
            termios.TCSADRAIN,
            old_settings,
        )

        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()