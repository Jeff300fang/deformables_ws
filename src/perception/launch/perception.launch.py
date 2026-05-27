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

    tapnn_back = Node(
        package="perception",
        executable="tapnn_back",
        output="screen",
        parameters=[
            {
                "tapnn_checkpoint_path": tapnn_checkpoint
            }
        ]
    )

    front_fit_spline = Node(
        package="perception",
        executable="front_fit_spline",
        output="screen",
        parameters=[
            {
                "body_translation_x": 0.85,
                "body_translation_y": 0.0,
                "body_translation_z": 0.29,
            }
        ],
    )

    back_fit_spline = Node(
        package="perception",
        executable="back_fit_spline",
        output="screen",
        parameters=[
            {
                "body_translation_x": -0.9,
                "body_translation_y": 0.0,
                "body_translation_z": 0.29,
            }
        ],
    )

    rope_point_joint = Node(
        package="perception",
        executable="rope_points_joint",
        output="screen",
    )

    perception_switch_monitor = Node(
        package='perception',
        executable="perception_switch_monitor",
        output="screen"
    )

    return LaunchDescription([
        front_node,
        back_node,
        tapnn_front,
        front_fit_spline,
        tapnn_back,
        back_fit_spline,
        rope_point_joint,
        perception_switch_monitor
    ])