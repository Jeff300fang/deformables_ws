#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from bkstools.bks_lib.bks_module import BKSModule, GripperWarning
from bkstools.bks_lib.bks_base import keep_communication_alive_sleep


HOST = "192.170.10.4"

GRIP_POS_UM = 28000
OPEN_POS_UM = 80000

MOVE_VEL_UMS = 25000


class LeftGripNode(Node):

    def __init__(self):
        super().__init__("left_grip_node")

        self.last_command = None
        self.busy = False

        self.get_logger().info("Connecting to BKS gripper...")
        self.bks = BKSModule(HOST)

        self.make_ready()

        self.subscription = self.create_subscription(
            Bool,
            "left_grip",
            self.left_grip_callback,
            1,
        )

        self.get_logger().info("Subscribed to /left_grip")

    def _sleep(self, duration):
        keep_communication_alive_sleep(self.bks, duration)

    def make_ready(self):
        try:
            self.get_logger().info("Making gripper ready...")
            self.bks.MakeReady()
            self._sleep(0.5)
            return True
        except Exception as e:
            self.get_logger().error(f"MakeReady failed: {e}")
            return False

    def report_status(self):
        sw = self.bks.plc_sync_input[0]

        if sw & self.bks.sw_error:
            self.get_logger().error(
                f"Gripper error: 0x{self.bks.err_code:04x}"
            )
            return "error"

        if sw & self.bks.sw_gripped:
            self.get_logger().info("Gripped object.")
            return "gripped"

        if sw & self.bks.sw_no_workpiece_detected:
            self.get_logger().warn("No workpiece detected.")
            return "no_workpiece"

        self.get_logger().info("Motion complete.")
        return "ok"

    def recover_if_error(self):
        try:
            sw = self.bks.plc_sync_input[0]

            if sw & self.bks.sw_error:
                self.get_logger().warn(
                    f"Recovering from gripper error: 0x{self.bks.err_code:04x}"
                )
                self.make_ready()
                return True

            return False

        except Exception as e:
            self.get_logger().warn(f"Could not check/recover error state: {e}")
            self.make_ready()
            return True

    def move_gripper(self, position_um):
        if self.busy:
            self.get_logger().warn("Gripper is busy; command ignored.")
            return

        self.busy = True

        try:
            self.recover_if_error()

            self.get_logger().info(
                f"Moving gripper to position: {position_um} µm"
            )

            self.bks.move_to_absolute_position(
                position_um,
                MOVE_VEL_UMS,
            )

            self._sleep(2.0)

            status = self.report_status()

            if status == "error":
                self.recover_if_error()

        except GripperWarning as e:
            self.get_logger().warn(f"BKS warning during move: {e}")

            try:
                self.report_status()
            except Exception as status_e:
                self.get_logger().warn(
                    f"Could not read status after warning: {status_e}"
                )

            self.recover_if_error()

        except Exception as e:
            self.get_logger().error(f"Gripper command failed: {e}")
            self.recover_if_error()

        finally:
            self.busy = False

    def left_grip_callback(self, msg: Bool):
        target = GRIP_POS_UM if msg.data else OPEN_POS_UM

        if msg.data:
            self.get_logger().info("Received TRUE -> grip position")
        else:
            self.get_logger().info("Received FALSE -> open position")

        self.move_gripper(target)


def main(args=None):
    rclpy.init(args=args)

    node = LeftGripNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()