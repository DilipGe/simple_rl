#!/usr/bin/env python3
"""ROS2 node: detect a yellow object from a webcam and publish its 3D pose as a TF transform.

Pipeline:
  1. Reuse ``detect_yellow()`` from ``yellow_detection.py`` to get the object's pixel
     bounding box (x, y, w, h) in each webcam frame.
  2. Back-project the box centroid into a 3D point using a pinhole camera model.
     A monocular camera cannot measure depth directly, so depth Z is estimated from the
     object's *known real-world width*:  Z = fx * real_width / pixel_width.
     Then:  X = (u - cx) * Z / fx,   Y = (v - cy) * Z / fy.
     The result is expressed in a ROS camera *optical* frame (x-right, y-down, z-forward).
  3. Broadcast a TF2 transform ``camera_frame -> object_frame`` (published on /tf) and a
     ``geometry_msgs/PoseStamped`` on ``<object_frame>/pose`` for convenience.

Intrinsics: if you have a calibrated camera, set fx/fy/cx/cy directly via parameters.
Otherwise they are approximated from the horizontal field-of-view and image size.

Run (after sourcing ROS2):
    source /opt/ros/humble/setup.bash
    python3 scripts/yellow_tf_publisher.py
    # with parameters:
    python3 scripts/yellow_tf_publisher.py --ros-args \
        -p object_real_width:=0.057 -p horizontal_fov_deg:=60.0 -p show_window:=true

Inspect:
    ros2 topic echo /yellow_object/pose
    ros2 run tf2_ros tf2_echo camera_optical_frame yellow_object
    rviz2   # add TF + Pose displays, fixed frame = camera_optical_frame
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

# Reuse the detection algorithm from the sibling script (DRY).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yellow_detection import annotate, detect_yellow  # noqa: E402


class YellowTfPublisher(Node):
    """Detects a yellow object and broadcasts its pose as a TF transform + PoseStamped."""

    def __init__(self) -> None:
        super().__init__("yellow_tf_publisher")

        # --- parameters ---
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("object_real_width", 0.05)  # metres, real width of the object
        self.declare_parameter("horizontal_fov_deg", 60.0)  # used only if fx/fy not given
        self.declare_parameter("fx", 0.0)  # focal length px; 0 => derive from FOV
        self.declare_parameter("fy", 0.0)
        self.declare_parameter("cx", 0.0)  # principal point px; 0 => image centre
        self.declare_parameter("cy", 0.0)
        self.declare_parameter("camera_frame", "camera_optical_frame")
        self.declare_parameter("object_frame", "yellow_object")
        self.declare_parameter("min_area", 800.0)
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("show_window", False)

        self.object_real_width = self.get_parameter("object_real_width").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.object_frame = self.get_parameter("object_frame").value
        self.min_area = self.get_parameter("min_area").value
        self.show_window = self.get_parameter("show_window").value

        # --- camera ---
        cam_index = self.get_parameter("camera_index").value
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {cam_index}")

        # Read one frame to learn the resolution, then compute intrinsics.
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Could not read an initial frame from the webcam")
        self.height, self.width = frame.shape[:2]
        self._init_intrinsics()

        # --- publishers ---
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, f"{self.object_frame}/pose", 10)

        rate = self.get_parameter("publish_rate").value
        self.timer = self.create_timer(1.0 / rate, self.on_timer)

        self.get_logger().info(
            f"Yellow TF publisher started: {self.width}x{self.height}, "
            f"fx={self.fx:.1f} fy={self.fy:.1f} cx={self.cx:.1f} cy={self.cy:.1f}, "
            f"object_real_width={self.object_real_width} m, "
            f"frames: {self.camera_frame} -> {self.object_frame}"
        )

    def _init_intrinsics(self) -> None:
        """Use supplied intrinsics, else approximate them from the horizontal FOV."""
        fx = self.get_parameter("fx").value
        fy = self.get_parameter("fy").value
        cx = self.get_parameter("cx").value
        cy = self.get_parameter("cy").value

        if fx <= 0.0:
            hfov = math.radians(self.get_parameter("horizontal_fov_deg").value)
            fx = (self.width / 2.0) / math.tan(hfov / 2.0)
        if fy <= 0.0:
            fy = fx  # assume square pixels
        if cx <= 0.0:
            cx = self.width / 2.0
        if cy <= 0.0:
            cy = self.height / 2.0

        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def on_timer(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("Failed to read frame from camera", throttle_duration_sec=2.0)
            return

        _, rois = detect_yellow(frame, self.min_area)

        if rois:
            # Track the largest blob (by box area) as the object of interest.
            x, y, w, h = max(rois, key=lambda r: r[2] * r[3])
            u = x + w / 2.0  # centroid pixel coordinates
            v = y + h / 2.0

            # Monocular depth from known object width, then pinhole back-projection.
            z = self.fx * self.object_real_width / float(w)
            px = (u - self.cx) * z / self.fx
            py = (v - self.cy) * z / self.fy

            self.publish_transform(px, py, z)
            self.publish_pose(px, py, z)

        if self.show_window:
            cv2.imshow("Yellow TF publisher", annotate(frame, rois))
            cv2.waitKey(1)

    def publish_transform(self, x: float, y: float, z: float) -> None:
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.camera_frame
        t.child_frame_id = self.object_frame
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)
        # Orientation is unknown for a point detection -> identity quaternion.
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def publish_pose(self, x: float, y: float, z: float) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0
        self.pose_pub.publish(msg)

    def destroy_node(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = YellowTfPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # On Ctrl+C, rclpy's signal handler may have already shut down the context;
        # calling shutdown() again raises RCLError. Guard against the double-shutdown.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
