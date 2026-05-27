#!/usr/bin/env python3

import argparse

import cv2
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="Path to rosbag2 folder")
    parser.add_argument("--topic", required=True, help="Color image topic")
    parser.add_argument("--output", default="output.mp4", help="Output mp4 file")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Only save every Nth image message",
    )
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    bridge = CvBridge()

    storage_options = rosbag2_py.StorageOptions(
        uri=args.bag,
        storage_id="sqlite3",
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    if args.topic not in type_map:
        print("Available topics:")
        for name, typ in type_map.items():
            print(f"  {name}: {typ}")
        raise RuntimeError(f"Topic not found: {args.topic}")

    if type_map[args.topic] != "sensor_msgs/msg/Image":
        raise RuntimeError(
            f"Topic {args.topic} has type {type_map[args.topic]}, "
            "but this script expects sensor_msgs/msg/Image"
        )

    writer = None
    seen_frames = 0
    saved_frames = 0

    print(f"Reading bag: {args.bag}")
    print(f"Using topic: {args.topic}")
    print(f"Saving every {args.stride} frame(s)")
    print(f"Output: {args.output}")

    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        if topic != args.topic:
            continue

        seen_frames += 1

        if seen_frames % args.stride != 0:
            continue

        msg = deserialize_message(data, Image)

        try:
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            print(f"Skipping frame {seen_frames} due to conversion error: {e}")
            continue

        if writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                args.output,
                fourcc,
                args.fps,
                (width, height),
            )

            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {args.output}")

            print(f"Video size: {width}x{height}")

        writer.write(frame)
        saved_frames += 1

        if saved_frames % 100 == 0:
            print(f"Wrote {saved_frames} frames")

    if writer is not None:
        writer.release()

    print("Done.")
    print(f"Image messages seen: {seen_frames}")
    print(f"Frames saved: {saved_frames}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()