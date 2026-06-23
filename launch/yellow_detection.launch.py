#!/usr/bin/env python3
"""Launch the yellow-object TF publisher (and optionally RViz) from one command.

This is a standalone script rather than an installed ROS2 package, so the node is
started with ExecuteProcess (running ``python3 scripts/yellow_tf_publisher.py``)
instead of the usual ``Node(package=..., executable=...)``.

Usage:
    source /opt/ros/humble/setup.bash
    ros2 launch launch/yellow_detection.launch.py
    # override parameters:
    ros2 launch launch/yellow_detection.launch.py object_real_width:=0.057 show_window:=true
    # also open RViz:
    ros2 launch launch/yellow_detection.launch.py rviz:=true

List all arguments:
    ros2 launch launch/yellow_detection.launch.py --show-args
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

# Resolve paths relative to this launch file so it works from any working directory.
_LAUNCH_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_LAUNCH_DIR)
_NODE_SCRIPT = os.path.join(_PROJECT_ROOT, "scripts", "yellow_tf_publisher.py")


def generate_launch_description() -> LaunchDescription:
    # --- declared launch arguments (all overridable on the command line) ---
    args = [
        DeclareLaunchArgument("camera_index", default_value="0", description="Webcam index."),
        DeclareLaunchArgument("object_real_width", default_value="0.05", description="Real object width (m)."),
        DeclareLaunchArgument("horizontal_fov_deg", default_value="60.0", description="Camera horizontal FOV (deg)."),
        DeclareLaunchArgument("camera_frame", default_value="camera_optical_frame", description="Parent TF frame."),
        DeclareLaunchArgument("object_frame", default_value="yellow_object", description="Child TF frame."),
        DeclareLaunchArgument("min_area", default_value="800.0", description="Min blob area (px)."),
        DeclareLaunchArgument("publish_rate", default_value="30.0", description="Publish rate (Hz)."),
        DeclareLaunchArgument("show_window", default_value="false", description="Show OpenCV preview window."),
        DeclareLaunchArgument("rviz", default_value="false", description="Also launch RViz2."),
    ]

    # --- the detection / TF publisher node ---
    yellow_node = ExecuteProcess(
        cmd=[
            "python3", _NODE_SCRIPT,
            "--ros-args",
            "-p", ["camera_index:=", LaunchConfiguration("camera_index")],
            "-p", ["object_real_width:=", LaunchConfiguration("object_real_width")],
            "-p", ["horizontal_fov_deg:=", LaunchConfiguration("horizontal_fov_deg")],
            "-p", ["camera_frame:=", LaunchConfiguration("camera_frame")],
            "-p", ["object_frame:=", LaunchConfiguration("object_frame")],
            "-p", ["min_area:=", LaunchConfiguration("min_area")],
            "-p", ["publish_rate:=", LaunchConfiguration("publish_rate")],
            "-p", ["show_window:=", LaunchConfiguration("show_window")],
        ],
        output="screen",
    )

    # --- optional RViz2 (only when rviz:=true) ---
    rviz_node = ExecuteProcess(
        cmd=["rviz2"],
        output="screen",
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription([*args, yellow_node, rviz_node])
