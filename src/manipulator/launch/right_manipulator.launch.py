#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    right_grip_node = Node(
        package="manipulator",
        executable="right_grip",
        name="right_grip_node",
        output="screen",
    )

    right_get_end_effector_pose = Node(
        package="manipulator",
        executable="right_get_end_effector_pose",
        name="right_iiwa_end_effector_pose_publisher",
        output="screen",
    )

    right_move_to_pose = Node(
        package="manipulator",
        executable="right_move_to_pose",
        name="right_iiwa_mpc_goal_pose_node",
        output="screen",
    )

    right_arm_driver = Node(
        package="manipulator",
        executable="right_arm_driver",
        name="right_safe_iiwa_joint_executor",
        output="screen"
    )
    

    return LaunchDescription([
        right_grip_node,
        right_get_end_effector_pose,
        right_move_to_pose,
        right_arm_driver,
    ])