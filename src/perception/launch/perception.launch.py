from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    perception_share = get_package_share_directory("perception")

    sam_checkpoint = os.path.join(
        perception_share,
        "checkpoints",
        "sam3.pt",
    )

    front_node = Node(
        package="perception",
        executable="sam_detection_front",
        name="sam_detection_front",
        output="screen",
        parameters=[
            {
                "sam_checkpoint_path": sam_checkpoint,
            }
        ],
    )

    back_node = Node(
        package="perception",
        executable="sam_detection_back",
        name="sam_detection_back",
        output="screen",
        parameters=[
            {
                "sam_checkpoint_path": sam_checkpoint,
            }
        ],
    )

    tapnn_front = Node(
        package="perception",
        executable="tapnn_front",
        output="screen",
    )

    return LaunchDescription([
        front_node,
        back_node,
        tapnn_front,
    ])