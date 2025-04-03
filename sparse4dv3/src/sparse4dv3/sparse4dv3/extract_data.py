#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import rclpy
from cv_bridge import CvBridge
import rosbag2_py
import rclpy.serialization
from sensor_msgs.msg import Image

def main():
    parser = argparse.ArgumentParser(
        description="Extract images from CAM_FRONT_LEFT/image_rect and CAM_FRONT_RIGHT/image_rect topics in a ROS2 bag file at a given timestamp."
    )
    parser.add_argument(
        '--bag_file', type=str, required=True,
        help="Path to the ROS2 bag file directory (e.g., /data/sparse4dv3/data/NuScenes-v1.0-mini-scene-0061)"
    )
    parser.add_argument(
        '--timestamp', type=float, required=True,
        help="Target timestamp in seconds (e.g., 1532402932.647284000)"
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help="Directory to save the extracted images (e.g., /data/sparse4dv3/output)"
    )
    parser.add_argument(
        '--time_tolerance', type=float, default=0.1,
        help="Time tolerance in seconds (default: 0.1). A tolerance of 0 might miss near matches."
    )
    args = parser.parse_args()

    bag_file = args.bag_file
    timestamp = args.timestamp
    output_dir = args.output_dir
    tolerance = args.time_tolerance

    # Define the target topics explicitly.
    target_topics = ['CAM_FRONT_LEFT/image_rect', 'CAM_FRONT_RIGHT/image_rect']

    # Convert timestamp and tolerance to nanoseconds for comparison.
    target_ns = int(timestamp * 1e9)
    tolerance_ns = int(tolerance * 1e9)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Looking for image messages in topics {target_topics} within ±{tolerance} s of timestamp {timestamp}...")

    # Initialize rclpy (without spinning a node)
    rclpy.init(args=sys.argv)

    # Set up the rosbag2_py SequentialReader
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_file, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader.open(storage_options, converter_options)

    bridge = CvBridge()
    # Track if messages were found for each target topic.
    found_topics = {target: False for target in target_topics}

    # Iterate over bag messages.
    while reader.has_next():
        topic, data, t = reader.read_next()  # t is in nanoseconds
        if abs(t - target_ns) <= tolerance_ns:
            for target in target_topics:
                if target in topic:
                    try:
                        # Deserialize the image message.
                        image_msg = rclpy.serialization.deserialize_message(data, Image)
                        # Convert the ROS image message to an OpenCV image (assuming 'bgr8' encoding).
                        cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
                        # Prepare a safe filename using the topic name and timestamp.
                        safe_topic = topic.strip('/').replace('/', '_')
                        filename = os.path.join(output_dir, f"{safe_topic}_{t}.png")
                        cv2.imwrite(filename, cv_image)
                        print(f"Saved image from topic '{topic}' at timestamp {t} ns to: {filename}")
                        found_topics[target] = True
                    except Exception as e:
                        print(f"Error processing image on topic '{topic}': {e}", file=sys.stderr)

    # Report any target topics for which no image was found.
    for target, found in found_topics.items():
        if not found:
            print(f"No image message found for topic containing '{target}' within the specified time tolerance.")

    rclpy.shutdown()

if __name__ == '__main__':
    main()
