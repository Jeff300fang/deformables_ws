import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

class PerceptionSwitchMonitor(Node):
    def __init__(self):
        super().__init__('perception_switch_monitor')
        self.create_subscription(
            PoseStamped,
            '/left/workstation/end_effector_pose',
            self.left_callback,
            1
        )
        self.create_subscription(
            PoseStamped,
            '/right/workstation/end_effector_pose',
            self.right_callback,
            1
        )

        self.threshold = 0.07

        self.left_pose = None
        self.right_pose = None
        self.sent = False
        self.switch_pub = self.create_publisher(
            Bool,
            '/change_view',
            1
        )
    
    def left_callback(self, msg):
        self.left_pose = msg
        self.update()

    def right_callback(self, msg):
        self.right_pose = msg
        self.update()

    def update(self):
        if self.left_pose is None or self.right_pose is None:
            self.get_logger().warn("Skipping because either left pose or right pose is None")
            return
        
        if self.left_pose.pose.position.x < self.threshold and self.right_pose.pose.position.x < self.threshold and not self.sent:
            self.switch_pub.publish(Bool(data=True))
            self.sent = True

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionSwitchMonitor()
    rclpy.spin(node)
