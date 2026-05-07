#!/usr/bin/env python3

import os
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


class EnterImageSaverWithArucoDebug(Node):
    def __init__(self):
        super().__init__("enter_image_saver_with_aruco_debug")

        # --------------------------------------------------
        # Input topics
        # --------------------------------------------------
        self.image_topic_1 = "/front_camera/color/image_raw"
        self.image_topic_2 = "/back_camera/color/image_raw"

        self.use_compressed = False  # True if subscribing to CompressedImage topics

        # --------------------------------------------------
        # Save folders
        # --------------------------------------------------
        self.folder_1 = os.path.expanduser("~/cam1")
        self.folder_2 = os.path.expanduser("~/cam2")

        os.makedirs(self.folder_1, exist_ok=True)
        os.makedirs(self.folder_2, exist_ok=True)

        # --------------------------------------------------
        # Debug output topics
        # --------------------------------------------------
        self.debug_topic_1 = "/cam1/aruco_debug"
        self.debug_topic_2 = "/cam2/aruco_debug"

        # --------------------------------------------------
        # ArUco settings
        # --------------------------------------------------
        self.aruco_dict_name = "4X4_1000"
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_1000
        )
        self.aruco_params = cv2.aruco.DetectorParameters()

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.use_new_aruco_api = True
            self.aruco_detector = cv2.aruco.ArucoDetector(
                self.aruco_dict,
                self.aruco_params,
            )
        else:
            self.use_new_aruco_api = False
            self.aruco_detector = None

        # --------------------------------------------------
        # ROS setup
        # --------------------------------------------------
        self.bridge = CvBridge()

        self.latest_img_1 = None
        self.latest_img_2 = None
        self.lock = threading.Lock()
        self.counter = 1

        msg_type = CompressedImage if self.use_compressed else Image

        self.sub1 = self.create_subscription(
            msg_type,
            self.image_topic_1,
            self.image_callback_1,
            10,
        )

        self.sub2 = self.create_subscription(
            msg_type,
            self.image_topic_2,
            self.image_callback_2,
            10,
        )

        self.debug_pub_1 = self.create_publisher(
            Image,
            self.debug_topic_1,
            10,
        )

        self.debug_pub_2 = self.create_publisher(
            Image,
            self.debug_topic_2,
            10,
        )

        self.get_logger().info(f"Subscribed to {self.image_topic_1}")
        self.get_logger().info(f"Subscribed to {self.image_topic_2}")
        self.get_logger().info(f"Publishing debug image to {self.debug_topic_1}")
        self.get_logger().info(f"Publishing debug image to {self.debug_topic_2}")
        self.get_logger().info(f"Using ArUco dictionary: {self.aruco_dict_name}")
        self.get_logger().info("Press Enter in this terminal to save both latest images.")

        self.input_thread = threading.Thread(
            target=self.wait_for_enter,
            daemon=True,
        )
        self.input_thread.start()

    def ros_img_to_cv2(self, msg):
        if self.use_compressed:
            return self.bridge.compressed_imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )

        return self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8",
        )

    def detect_aruco_debug(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.use_new_aruco_api:
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params,
            )

        debug_img = img.copy()

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)
            ids_flat = ids.flatten()

            for marker_corners, marker_id in zip(corners, ids_flat):
                pts = marker_corners.reshape(-1, 2)
                center = np.mean(pts, axis=0).astype(int)

                cv2.putText(
                    debug_img,
                    f"id:{int(marker_id)}",
                    tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            label = f"Detected {len(ids_flat)} ArUco markers"
            color = (0, 255, 0)

            return debug_img, len(ids_flat), ids_flat

        label = "No ArUco markers detected"
        color = (0, 0, 255)

        cv2.putText(
            debug_img,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

        return debug_img, 0, np.array([], dtype=int)

    def publish_debug_image(self, debug_pub, debug_img, header):
        debug_msg = self.bridge.cv2_to_imgmsg(
            debug_img,
            encoding="bgr8",
        )
        debug_msg.header = header
        debug_pub.publish(debug_msg)

    def image_callback_1(self, msg):
        try:
            img = self.ros_img_to_cv2(msg)
            debug_img, n, ids = self.detect_aruco_debug(img)

            self.publish_debug_image(
                self.debug_pub_1,
                debug_img,
                msg.header,
            )

            with self.lock:
                self.latest_img_1 = img

        except Exception as e:
            self.get_logger().error(f"Camera 1 callback failed: {e}")

    def image_callback_2(self, msg):
        try:
            img = self.ros_img_to_cv2(msg)
            debug_img, n, ids = self.detect_aruco_debug(img)

            self.publish_debug_image(
                self.debug_pub_2,
                debug_img,
                msg.header,
            )

            with self.lock:
                self.latest_img_2 = img

        except Exception as e:
            self.get_logger().error(f"Camera 2 callback failed: {e}")

    def wait_for_enter(self):
        while rclpy.ok():
            try:
                input()
                self.save_images()
            except EOFError:
                break

    def save_images(self):
        with self.lock:
            if self.latest_img_1 is None or self.latest_img_2 is None:
                self.get_logger().warn("Have not received both images yet.")
                return

            img1 = self.latest_img_1.copy()
            img2 = self.latest_img_2.copy()

        filename = f"image{self.counter:02d}.jpg"

        path1 = os.path.join(self.folder_1, filename)
        path2 = os.path.join(self.folder_2, filename)

        ok1 = cv2.imwrite(path1, img1)
        ok2 = cv2.imwrite(path2, img2)

        if ok1:
            self.get_logger().info(f"Saved {path1}")
        else:
            self.get_logger().error(f"Failed to save {path1}")

        if ok2:
            self.get_logger().info(f"Saved {path2}")
        else:
            self.get_logger().error(f"Failed to save {path2}")

        if ok1 and ok2:
            self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = EnterImageSaverWithArucoDebug()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()