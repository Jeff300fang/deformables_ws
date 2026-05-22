#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    left_grip_node = Node(
        package="manipulator",
        executable="left_grip",
        name="left_grip_node",
        output="screen",
    )

    right_grip_node = Node(
        package="manipulator",
        executable="right_grip",
        name="right_grip_node",
        output="screen",
    )

    return LaunchDescription([
        left_grip_node,
        right_grip_node,
    ])