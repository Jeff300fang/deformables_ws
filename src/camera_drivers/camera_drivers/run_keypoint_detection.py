#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import torch
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import (
    Image,
    CompressedImage,
    CameraInfo,
    PointCloud2,
)
import sensor_msgs_py.point_cloud2 as pc2

from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

from tapnet.tapnext.tapnext_torch import TAPNext


CKPT_SIZE = (256, 256)


class TAPNextManualKeypointDepthPointCloudNode(Node):
    def __init__(self):
        super().__init__("tapnext_manual_keypoint_depth_pointcloud_node")

        # --------------------------------------------------
        # Topics
        # --------------------------------------------------

        # Orbbec / front camera
        self.image_topic = "/back_camera/color/image_raw"
        self.depth_topic = "/back_camera/depth/image_raw"
        self.camera_info_topic = "/back_camera/color/camera_info"

        # RealSense aligned depth version
        # self.image_topic = "/realsense/camera/color/image_raw"
        # self.depth_topic = "/realsense/camera/aligned_depth_to_color/image_raw"
        # self.camera_info_topic = "/realsense/camera/color/camera_info"

        self.annotated_topic = "/tapnext/annotated_image"
        self.tracks_topic = "/tapnext/tracks"
        self.pointcloud_topic = "/tapnext/points_3d"

        self.ckpt_path = "/home/jeffreyfang/deformables/src/tapnet/checkpoints/tapnextpp_ckpt.pt"

        self.bridge = CvBridge()

        # --------------------------------------------------
        # Camera intrinsics
        # --------------------------------------------------
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # --------------------------------------------------
        # Device
        # --------------------------------------------------
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.get_logger().info(f"Using device: {self.device}")

        # --------------------------------------------------
        # Load TAPNext
        # --------------------------------------------------
        self.model = TAPNext(image_size=CKPT_SIZE)

        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        self.model.load_state_dict({
            k.replace("tapnext.", ""): v
            for k, v in ckpt["state_dict"].items()
        })

        self.model = self.model.to(self.device).eval()

        # --------------------------------------------------
        # Runtime tracking state
        # --------------------------------------------------
        self.initialized = False
        self.tracking_state = None
        self.query_points_np_orig = None
        self.frame_idx = 0

        # --------------------------------------------------
        # ROS interfaces
        # --------------------------------------------------
        self.image_sub = Subscriber(self, Image, self.image_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub],
            queue_size=10,
            slop=0.05,
        )
        self.sync.registerCallback(self.image_depth_callback)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.annotated_pub = self.create_publisher(
            CompressedImage,
            self.annotated_topic + "/compressed",
            10,
        )

        self.tracks_pub = self.create_publisher(
            Detection2DArray,
            self.tracks_topic,
            10,
        )

        self.pointcloud_pub = self.create_publisher(
            PointCloud2,
            self.pointcloud_topic,
            10,
        )

        self.get_logger().info(f"Subscribed to RGB:         {self.image_topic}")
        self.get_logger().info(f"Subscribed to depth:       {self.depth_topic}")
        self.get_logger().info(f"Subscribed to camera info: {self.camera_info_topic}")
        self.get_logger().info(f"Publishing annotated:      {self.annotated_topic}/compressed")
        self.get_logger().info(f"Publishing tracks:         {self.tracks_topic}")
        self.get_logger().info(f"Publishing point cloud:    {self.pointcloud_topic}")

    # ------------------------------------------------------------------
    # Camera info
    # ------------------------------------------------------------------
    def camera_info_callback(self, msg: CameraInfo):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    def crop_and_downscale(self, img_rgb):
        h, w = img_rgb.shape[:2]

        if w >= h:
            startx = w // 2 - h // 2
            starty = 0
            crop_size = h
        else:
            startx = 0
            starty = h // 2 - w // 2
            crop_size = w

        cropped = img_rgb[
            starty:starty + crop_size,
            startx:startx + crop_size,
        ]

        resized = cv.resize(
            cropped,
            (CKPT_SIZE[1], CKPT_SIZE[0]),
            interpolation=cv.INTER_AREA,
        )

        return cropped, resized, (startx, starty, crop_size)

    def crop_depth_with_rgb_crop(self, depth_img, crop_info):
        startx, starty, crop_size = crop_info

        return depth_img[
            starty:starty + crop_size,
            startx:startx + crop_size,
        ]

    def preprocess_frame(self, frame_rgb):
        cropped_rgb, resized_rgb, crop_info = self.crop_and_downscale(frame_rgb)

        # --------------------------------------------------
        # NEW: image preprocessing for stable keypoints
        # --------------------------------------------------

        # 1. Slight blur (reduces high-frequency noise)
        resized_rgb = cv.GaussianBlur(resized_rgb, (5, 5), 0)

        # 2. CLAHE contrast normalization (very important)
        lab = cv.cvtColor(resized_rgb, cv.COLOR_RGB2LAB)
        l, a, b = cv.split(lab)

        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        resized_rgb = cv.merge((l, a, b))
        resized_rgb = cv.cvtColor(resized_rgb, cv.COLOR_LAB2RGB)

        # 3. Optional sharpening (helps rope edges)
        kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
        resized_rgb = cv.filter2D(resized_rgb, -1, kernel)

        # --------------------------------------------------

        video_np = resized_rgb[None, None].astype(np.float32)
        video = torch.from_numpy(video_np)
        video = (video / 255.0) * 2.0 - 1.0
        video = video.to(self.device)

        return cropped_rgb, resized_rgb, video, crop_info

    # ------------------------------------------------------------------
    # Manual query point selection
    # ------------------------------------------------------------------
    def select_query_points_manual(self, frame_rgb):
        frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        display = frame_bgr.copy()
        points_xy = []

        window_name = "Select TAPNext query points"

        def redraw():
            nonlocal display
            display = frame_bgr.copy()

            for i, (x, y) in enumerate(points_xy):
                cv.circle(display, (x, y), 5, (0, 255, 0), -1)
                cv.putText(
                    display,
                    str(i),
                    (x + 6, y - 6),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            cv.putText(
                display,
                "Left click: add | u: undo | c/Enter: confirm | q/Esc: quit",
                (20, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        def mouse_callback(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                points_xy.append((x, y))
                redraw()

        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.setMouseCallback(window_name, mouse_callback)

        redraw()

        while rclpy.ok():
            cv.imshow(window_name, display)
            key = cv.waitKey(20) & 0xFF

            if key == ord("u"):
                if points_xy:
                    points_xy.pop()
                    redraw()

            elif key in (ord("c"), 13):
                break

            elif key in (ord("q"), 27):
                points_xy = []
                break

        cv.destroyWindow(window_name)

        if len(points_xy) == 0:
            return None

        query_points = np.zeros((len(points_xy), 3), dtype=np.float32)
        query_points[:, 0] = 0.0
        query_points[:, 1] = np.array([p[1] for p in points_xy], dtype=np.float32)
        query_points[:, 2] = np.array([p[0] for p in points_xy], dtype=np.float32)

        return query_points

    # ------------------------------------------------------------------
    # TAPNext tracking
    # ------------------------------------------------------------------
    def initialize_tracker(self, video, crop_h):
        model_h, model_w = CKPT_SIZE

        scale_x = model_w / crop_h
        scale_y = model_h / crop_h

        query_points_np = self.query_points_np_orig.copy()
        query_points_np[:, 1] *= scale_y
        query_points_np[:, 2] *= scale_x
        query_points_np = query_points_np[None]

        query_points = torch.from_numpy(query_points_np).to(self.device)

        with torch.no_grad():
            pred_tracks, _, visible_logits, self.tracking_state = self.model(
                video=video,
                query_points=query_points,
            )

        tracks = pred_tracks.cpu().numpy()[0, 0]
        visible = (visible_logits > 0).cpu().numpy()[0, 0, :, 0]

        self.initialized = True

        self.get_logger().info(
            f"Initialized TAPNext with {query_points_np.shape[1]} manually selected points"
        )

        return tracks, visible

    def step_tracker(self, video):
        with torch.no_grad():
            pred_tracks, _, visible_logits, self.tracking_state = self.model(
                video=video,
                state=self.tracking_state,
            )

        tracks = pred_tracks.cpu().numpy()[0, 0]
        visible = (visible_logits > 0).cpu().numpy()[0, 0, :, 0]

        return tracks, visible

    # ------------------------------------------------------------------
    # Depth querying
    # ------------------------------------------------------------------
    def query_depths_at_tracks(self, cropped_depth, tracks_xy, visible):
        h, w = cropped_depth.shape[:2]
        depths_m = []

        for xy, is_visible in zip(tracks_xy, visible):
            if not is_visible:
                depths_m.append(np.nan)
                continue

            x = int(round(float(xy[0])))
            y = int(round(float(xy[1])))

            if x < 0 or x >= w or y < 0 or y >= h:
                depths_m.append(np.nan)
                continue

            # ---------------------------------------
            # NEW: 3x3 neighborhood min depth sampling
            # ---------------------------------------
            x0 = max(0, x - 1)
            x1 = min(w, x + 2)
            y0 = max(0, y - 1)
            y1 = min(h, y + 2)

            patch = cropped_depth[y0:y1, x0:x1]

            if patch.size == 0:
                depths_m.append(np.nan)
                continue

            if cropped_depth.dtype == np.uint16:
                patch_m = patch.astype(np.float32) * 0.001
            else:
                patch_m = patch.astype(np.float32)

            # Remove invalid values
            valid = np.isfinite(patch_m) & (patch_m > 0.0)

            if not np.any(valid):
                depths_m.append(np.nan)
                continue

            # Take minimum valid depth (closest point)
            d_m = np.min(patch_m[valid])
            depths_m.append(d_m)
            # ---------------------------------------

        return np.asarray(depths_m, dtype=np.float32)

    # ------------------------------------------------------------------
    # 3D projection + point cloud publishing
    # ------------------------------------------------------------------
    def tracks_to_3d_points(self, tracks_xy, visible, depths_m, crop_info):
        if self.fx is None:
            return []

        startx, starty, _ = crop_info
        points = []

        for xy, is_visible, z in zip(tracks_xy, visible, depths_m):
            if not is_visible:
                continue
            if not np.isfinite(z) or z <= 0.0:
                continue

            # pixel → full image coords
            u = float(xy[0]) + float(startx)
            v = float(xy[1]) + float(starty)

            # ---------------------------
            # Camera optical coordinates
            # ---------------------------
            x_c = (u - self.cx) * z / self.fx
            y_c = (v - self.cy) * z / self.fy
            z_c = float(z)

            # ---------------------------
            # Transform to body frame (FLU)
            # ---------------------------
            x_b = z_c          # forward
            y_b = -x_c         # left
            z_b = -y_c         # up

            points.append((x_b, y_b, z_b))

        return points

    def publish_pointcloud(self, msg_header, tracks_xy, visible, depths_m, crop_info):
        points = self.tracks_to_3d_points(
            tracks_xy,
            visible,
            depths_m,
            crop_info,
        )

        cloud_msg = pc2.create_cloud_xyz32(msg_header, points)

        # IMPORTANT: change frame_id
        cloud_msg.header.frame_id = "base_link"  # or your body frame

        self.pointcloud_pub.publish(cloud_msg)

    # ------------------------------------------------------------------
    # ROS publishing
    # ------------------------------------------------------------------
    def publish_tracks(self, msg_header, tracks_xy, visible, depths_m):
        out = Detection2DArray()
        out.header = msg_header

        for i, (xy, is_visible, d_m) in enumerate(zip(tracks_xy, visible, depths_m)):
            if not is_visible:
                continue

            x, y = float(xy[0]), float(xy[1])

            det = Detection2D()
            det.header = msg_header
            det.bbox.center.position.x = x
            det.bbox.center.position.y = y
            det.bbox.size_x = 4.0
            det.bbox.size_y = 4.0

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = f"tap_point_{i}"

            if np.isfinite(d_m):
                hyp.hypothesis.score = float(d_m)
            else:
                hyp.hypothesis.score = -1.0

            det.results.append(hyp)
            out.detections.append(det)

        self.tracks_pub.publish(out)

    def draw_tracks_with_depth(self, cropped_rgb, tracks_xy, visible, depths_m):
        vis_img = cropped_rgb.copy()

        for i, (xy, is_visible, d_m) in enumerate(zip(tracks_xy, visible, depths_m)):
            if not is_visible:
                continue

            x = int(round(float(xy[0])))
            y = int(round(float(xy[1])))

            cv.circle(vis_img, (x, y), 5, (0, 255, 0), -1)

            if np.isfinite(d_m):
                label = f"{i}: {d_m:.3f} m"
            else:
                label = f"{i}: no depth"

            cv.putText(
                vis_img,
                label,
                (x + 6, y - 6),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        return vis_img

    # ------------------------------------------------------------------
    # Main synced RGB-depth callback
    # ------------------------------------------------------------------
    def image_depth_callback(self, rgb_msg, depth_msg):
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(
                rgb_msg,
                desired_encoding="bgr8",
            )
            frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)

            depth_img = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )
            depth_img = np.asarray(depth_img)

            cropped_rgb, resized_rgb, video, crop_info = self.preprocess_frame(frame_rgb)
            cropped_depth = self.crop_depth_with_rgb_crop(depth_img, crop_info)

            crop_h, crop_w = cropped_rgb.shape[:2]

            if not self.initialized:
                self.get_logger().info(
                    "First frame received. Select query points in the OpenCV window."
                )

                qp = self.select_query_points_manual(cropped_rgb)

                if qp is None:
                    self.get_logger().warn(
                        "No query points selected. Waiting for another frame."
                    )
                    return

                self.query_points_np_orig = qp
                tracks_yx, visible = self.initialize_tracker(video, crop_h)

            else:
                tracks_yx, visible = self.step_tracker(video)

            # TAPNext output convention from your existing script.
            tracks_xy = tracks_yx[:, ::-1]

            # Scale from 256x256 model coordinates back to cropped image coordinates.
            model_h, model_w = CKPT_SIZE
            tracks_xy[:, 0] *= crop_w / model_w
            tracks_xy[:, 1] *= crop_h / model_h

            depths_m = self.query_depths_at_tracks(
                cropped_depth,
                tracks_xy,
                visible,
            )

            self.publish_tracks(
                rgb_msg.header,
                tracks_xy,
                visible,
                depths_m,
            )

            self.publish_pointcloud(
                rgb_msg.header,
                tracks_xy,
                visible,
                depths_m,
                crop_info,
            )

            annotated_rgb = self.draw_tracks_with_depth(
                cropped_rgb,
                tracks_xy,
                visible,
                depths_m,
            )

            annotated_bgr = cv.cvtColor(annotated_rgb, cv.COLOR_RGB2BGR)

            ok, encoded = cv.imencode(
                ".jpg",
                annotated_bgr,
                [int(cv.IMWRITE_JPEG_QUALITY), 90],
            )

            if not ok:
                self.get_logger().warn("Failed to encode annotated image as JPEG")
                return

            annotated_msg = CompressedImage()
            annotated_msg.header = rgb_msg.header
            annotated_msg.format = "jpeg"
            annotated_msg.data = encoded.tobytes()

            self.annotated_pub.publish(annotated_msg)

            self.frame_idx += 1

        except Exception as e:
            self.get_logger().error(f"TAPNext RGB/depth callback failed: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = TAPNextManualKeypointDepthPointCloudNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    cv.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()