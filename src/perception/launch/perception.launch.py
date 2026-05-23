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

    tapnn_checkpoint = os.path.join(
        perception_share,
        "checkpoints",
        "tapnextpp_ckpt.pt",
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
        parameters=[
            {
                "tapnn_checkpoint_path": tapnn_checkpoint
            }
        ]
    )

    fit_spline = Node(
        package="perception",
        executable="fit_spline",
        output="screen",
        parameters=[
            {
                "body_translation_x": 0.88,
                "body_translation_y": 0.0,
                "body_translation_z": 0.29,
            }
        ],
    )

    return LaunchDescription([
        front_node,
        back_node,
        tapnn_front,
        fit_spline
    ])