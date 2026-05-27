import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Bool

class RopePointJoint(Node):
    def __init__(self):
        super().__init__('joint_rope_command')
        
        self.create_subscription(
            PoseArray,
            '/front/rope_poses',
            self.front_rope_pose_callback,
            1,
        )

        self.create_subscription(
            PoseArray,
            '/back/rope_poses',
            self.back_rope_pose_callback,
            1
        )

        self.stop_front_pub = self.create_publisher(
            Bool,
            '/stop_front',
            1
        )

        self.rope_pub = self.create_publisher(
            PoseArray,
            '/rope_poses',
            1
        )

        self.front_rope = None
        self.back_rope = None
        self.stop_command_sent = False

    def front_rope_pose_callback(self, msg):
        self.front_rope = msg
        self.update(True, False)

    def back_rope_pose_callback(self, msg):
        self.back_rope = msg
        self.update(False, True)
    
    def update(self, front, back):
        if self.back_rope is not None and not self.stop_command_sent:
            self.stop_front_pub.publish(Bool(data=True))
            self.stop_command_sent = True
        if self.stop_command_sent and back:
            self.rope_pub.publish(self.back_rope)
        if not self.stop_command_sent and front:
            self.rope_pub.publish(self.front_rope)
    
def main(args=None):
    rclpy.init(args=args)
    node = RopePointJoint()
    rclpy.spin(node)

