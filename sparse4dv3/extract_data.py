# #!/usr/bin/env python3
# import os
# import cv2
# import copy
# import sys
# import time
# import yaml
# import rclpy
# from rclpy.node import Node
# from cv_bridge import CvBridge

# # Import ROS message types
# from sensor_msgs.msg import Image, CameraInfo
# from tf2_msgs.msg import TFMessage
# import rclpy.serialization

# # Import rosbag2_py API (make sure your ROS2 distro supports it)
# import rosbag2_py

# class BagExtractor(Node):
#     def __init__(self):
#         super().__init__('bag_extractor')
#         # Declare parameters (can be set via CLI or a YAML file)
#         self.declare_parameter('bag_file', '')
#         self.declare_parameter('timestamp', 0.0)      # Provided in seconds
#         self.declare_parameter('output_dir', 'output')
#         self.declare_parameter('time_tolerance', 0.1)   # in seconds
        
#         self.bag_file = self.get_parameter('bag_file').value
#         self.timestamp = self.get_parameter('timestamp').value
#         self.output_dir = self.get_parameter('output_dir').value
#         self.time_tolerance = self.get_parameter('time_tolerance').value
        
#         if not self.bag_file:
#             self.get_logger().error("No bag_file parameter provided!")
#             rclpy.shutdown()
#             return
        
#         # Convert provided timestamp and tolerance into nanoseconds
#         self.target_ns = int(self.timestamp * 1e9)
#         self.tolerance_ns = int(self.time_tolerance * 1e9)
        
#         # Initialize cv_bridge for image conversion
#         self.bridge = CvBridge()
        
#         # Create output directory if it does not exist
#         os.makedirs(self.output_dir, exist_ok=True)
        
#         self.get_logger().info(f"Starting extraction from {self.bag_file} at timestamp {self.timestamp}s")
#         self.extract_data_from_bag()

#     def extract_data_from_bag(self):
#         # Open the bag file using rosbag2_py API
#         reader = rosbag2_py.SequentialReader()
#         storage_options = rosbag2_py.StorageOptions(uri=self.bag_file, storage_id='sqlite3')
#         converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr',
#                                                         output_serialization_format='cdr')
#         reader.open(storage_options, converter_options)
        
#         # Containers for collected data
#         tf_data = []
#         # Dictionary to store camera_info messages by topic (one per camera)
#         camera_info_dict = {}
        
#         # Iterate over the bag file messages
#         while reader.has_next():
#             topic, data, t = reader.read_next()  # t is in nanoseconds
#             # Check if the message timestamp is within the tolerance of the provided timestamp
#             if abs(t - self.target_ns) <= self.tolerance_ns:
#                 # Process image topics (assumes topics containing 'image' in their name)
#                 if 'image' in topic:
#                     self.save_image(topic, data, t)
#                 # Process camera info topics (typically named like /cameraX/camera_info)
#                 elif 'camera_info' in topic:
#                     camera_info_dict[topic] = self.deserialize_camera_info(data)
#                     self.get_logger().info(f"Extracted camera info from {topic} at timestamp {t}")
#                 # Process TF topics
#                 elif topic in ['/tf', '/tf_static']:
#                     tf_data.extend(self.extract_tf(data))
        
#         # Dump camera info(s) to YAML file(s)
#         for cam_topic, info in camera_info_dict.items():
#             # Clean up topic name to use as file name
#             fname = cam_topic.strip('/').replace('/', '_') + '_camera_info.yaml'
#             yaml_path = os.path.join(self.output_dir, fname)
#             with open(yaml_path, 'w') as f:
#                 yaml.dump(info, f)
#             self.get_logger().info(f"Saved camera info for {cam_topic} to {yaml_path}")
        
#         # Dump TF data if available
#         if tf_data:
#             tf_yaml_path = os.path.join(self.output_dir, 'tf_data.yaml')
#             with open(tf_yaml_path, 'w') as f:
#                 yaml.dump(tf_data, f)
#             self.get_logger().info(f"Saved TF data to {tf_yaml_path}")
        
#         self.get_logger().info("Extraction complete.")
#         rclpy.shutdown()

#     def save_image(self, topic, data, timestamp):
#         """
#         Deserialize an Image message, convert it to a cv2 image, and save as PNG.
#         """
#         try:
#             image_msg = rclpy.serialization.deserialize_message(data, Image)
#             # Convert ROS image to OpenCV image (assuming bgr8 encoding)
#             cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
#         except Exception as e:
#             self.get_logger().error(f"Error processing image from {topic}: {e}")
#             return
        
#         # Create a subfolder for this camera based on its topic name
#         camera_dir = os.path.join(self.output_dir, topic.strip('/').replace('/', '_'))
#         os.makedirs(camera_dir, exist_ok=True)
#         # Use the bag timestamp as part of the filename
#         filename = os.path.join(camera_dir, f"{timestamp}.png")
#         cv2.imwrite(filename, cv_image)
#         self.get_logger().info(f"Saved image from {topic} at timestamp {timestamp} to {filename}")

#     def deserialize_camera_info(self, data):
#         """
#         Deserialize a CameraInfo message and convert it to a dictionary.
#         """
#         cam_info_msg = rclpy.serialization.deserialize_message(data, CameraInfo)
#         info_dict = {
#             'header': {
#                 'stamp': {
#                     'sec': cam_info_msg.header.stamp.sec,
#                     'nanosec': cam_info_msg.header.stamp.nanosec,
#                 },
#                 'frame_id': cam_info_msg.header.frame_id,
#             },
#             'height': cam_info_msg.height,
#             'width': cam_info_msg.width,
#             'distortion_model': cam_info_msg.distortion_model,
#             'd': list(cam_info_msg.d),
#             'k': list(cam_info_msg.k),
#             'r': list(cam_info_msg.r),
#             'p': list(cam_info_msg.p),
#             'binning_x': cam_info_msg.binning_x,
#             'binning_y': cam_info_msg.binning_y,
#             'roi': {
#                 'x_offset': cam_info_msg.roi.x_offset,
#                 'y_offset': cam_info_msg.roi.y_offset,
#                 'height': cam_info_msg.roi.height,
#                 'width': cam_info_msg.roi.width,
#                 'do_rectify': cam_info_msg.roi.do_rectify,
#             }
#         }
#         return info_dict

#     def extract_tf(self, data):
#         """
#         Deserialize a TFMessage and convert each TransformStamped into a dictionary.
#         """
#         tf_msg = rclpy.serialization.deserialize_message(data, TFMessage)
#         transforms_list = []
#         for transform in tf_msg.transforms:
#             t_data = {
#                 'header': {
#                     'stamp': {
#                         'sec': transform.header.stamp.sec,
#                         'nanosec': transform.header.stamp.nanosec,
#                     },
#                     'frame_id': transform.header.frame_id,
#                 },
#                 'child_frame_id': transform.child_frame_id,
#                 'transform': {
#                     'translation': {
#                         'x': transform.transform.translation.x,
#                         'y': transform.transform.translation.y,
#                         'z': transform.transform.translation.z,
#                     },
#                     'rotation': {
#                         'x': transform.transform.rotation.x,
#                         'y': transform.transform.rotation.y,
#                         'z': transform.transform.rotation.z,
#                         'w': transform.transform.rotation.w,
#                     }
#                 }
#             }
#             transforms_list.append(t_data)
#         return transforms_list

# def main(args=None):
#     rclpy.init(args=args)
#     node = BagExtractor()
#     executor = rclpy.executors.SingleThreadedExecutor(context=node.get_context())
#     executor.add_node(node)
#     try:
#         executor.spin()
#     finally:
#         executor.shutdown()
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()
