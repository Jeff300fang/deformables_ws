from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # =========================
        # Front Orbbec Gemini 336L
        # =========================
        Node(
            package="orbbec_camera",
            executable="orbbec_camera_node",
            namespace="front_camera",
            name="camera",
            output="screen",
            parameters=[{
                "serial_number": "CPCG853000L1",

                "enable_color": True,
                "enable_depth": True,
                "enable_ir": False,
                "enable_sync": True,

                # Depth-to-color alignment
                "depth_registration": True,

                # 🔧 REQUIRED: enforce compatible profiles
                "color_width": 640,
                "color_height": 480,
                "color_fps": 30,

                "depth_width": 640,
                "depth_height": 480,
                "depth_fps": 30,

                # Optional: force software alignment if HW fails
                # "align_mode": "SW",
            }],
        ),

        # =========================
        # Back Orbbec Gemini 336L
        # =========================
        Node(
            package="orbbec_camera",
            executable="orbbec_camera_node",
            namespace="back_camera",
            name="camera",
            output="screen",
            parameters=[{
                "serial_number": "CPCG853000JB",

                "enable_color": True,
                "enable_depth": True,
                "enable_ir": False,
                "enable_sync": True,

                "depth_registration": True,

                # 🔧 SAME profiles required
                "color_width": 640,
                "color_height": 480,
                "color_fps": 30,

                "depth_width": 640,
                "depth_height": 480,
                "depth_fps": 30,

                # "align_mode": "SW",
            }],
        ),

        # =========================
        # Intel RealSense
        # =========================
        Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            namespace="realsense",
            name="camera",
            output="screen",
            parameters=[{
                "enable_color": True,
                "enable_depth": True,
                "enable_infra1": False,
                "enable_infra2": False,

                # Depth-to-color alignment
                "align_depth.enable": True,

                # 🔧 ALSO constrain profiles (prevents USB overload)
                "rgb_camera.width": 640,
                "rgb_camera.height": 480,
                "rgb_camera.fps": 30,

                "depth_module.width": 640,
                "depth_module.height": 480,
                "depth_module.fps": 30,
            }],
        ),
    ])