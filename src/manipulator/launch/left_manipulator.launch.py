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

    left_get_end_effector_pose = Node(
        package="manipulator",
        executable="left_get_end_effector_pose",
        name="left_iiwa_end_effector_pose_publisher",
        output="screen",
    )

    left_move_to_pose = Node(
        package="manipulator",
        executable="left_move_to_pose",
        name="left_iiwa_mpc_goal_pose_node",
        output="screen",
    )

    left_arm_driver = Node(
        package="manipulator",
        executable="left_arm_driver",
        name="left_safe_iiwa_joint_executor",
        output="screen"
    )
    

    return LaunchDescription([
        left_grip_node,
        left_get_end_effector_pose,
        left_move_to_pose,
        left_arm_driver,
    ])