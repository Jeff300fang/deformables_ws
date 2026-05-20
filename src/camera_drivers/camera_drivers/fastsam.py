#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import FastSAM


class FastSamRopeNode(Node):
    def __init__(self):
        super().__init__("fast_sam_rope")

        self.declare_parameter("image_topic", "/front_camera/color/image_raw")
        self.declare_parameter("segmented_topic", "/fast_sam/rope_segmentation")
        self.declare_parameter("model_path", "FastSAM-s.pt")
        self.declare_parameter("text_prompt", "rope")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.2)
        self.declare_parameter("iou", 0.9)
        self.declare_parameter("device", "cuda")

        image_topic = self.get_parameter("image_topic").value
        segmented_topic = self.get_parameter("segmented_topic").value
        model_path = self.get_parameter("model_path").value

        self.text_prompt = self.get_parameter("text_prompt").value
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.device = self.get_parameter("device").value

        self.bridge = CvBridge()
        self.model = FastSAM(model_path)

        self.sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10,
        )

        self.pub = self.create_publisher(
            Image,
            segmented_topic,
            10,
        )

        self.get_logger().info(f"Subscribed to: {image_topic}")
        self.get_logger().info(f"Publishing to: {segmented_topic}")
        self.get_logger().info(f"FastSAM text prompt: {self.text_prompt}")

    def image_callback(self, msg: Image):
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            results = self.model.predict(
                source=frame_rgb,
                texts=[self.text_prompt],   # "rope"
                device=self.device,
                retina_masks=True,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )

            result = results[0]
            overlay_rgb = result.plot()
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

            out_msg = self.bridge.cv2_to_imgmsg(overlay_bgr, encoding="bgr8")
            out_msg.header = msg.header
            self.pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"FastSAM failed: {e}")

    def draw_masks(self, frame_bgr: np.ndarray, ann):
        overlay = frame_bgr.copy()

        if ann is None:
            return overlay

        # FastSAMPrompt can return different formats depending on version.
        if isinstance(ann, list):
            masks = ann
        else:
            masks = [ann]

        combined_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)

        for mask in masks:
            mask_np = np.asarray(mask)

            if mask_np.ndim == 3:
                mask_np = mask_np.squeeze()

            if mask_np.shape != frame_bgr.shape[:2]:
                mask_np = cv2.resize(
                    mask_np.astype(np.uint8),
                    (frame_bgr.shape[1], frame_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            combined_mask[mask_np > 0] = 255

        colored = np.zeros_like(frame_bgr)
        colored[:, :, 1] = 255  # green mask

        alpha = 0.45
        mask_bool = combined_mask > 0
        overlay[mask_bool] = cv2.addWeighted(
            frame_bgr[mask_bool],
            1.0 - alpha,
            colored[mask_bool],
            alpha,
            0,
        )

        contours, _ = cv2.findContours(
            combined_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        return overlay


def main(args=None):
    rclpy.init(args=args)
    node = FastSamRopeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()