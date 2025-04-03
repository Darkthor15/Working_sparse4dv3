#!/usr/bin/env python3
import os
import cv2
import sys
import time
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, RegionOfInterest
from geometry_msgs.msg import TransformStamped
import tf2_ros
from cv_bridge import CvBridge

class MultiCameraPublisher(Node):
    def __init__(self):
        super().__init__('multi_camera_publisher')
        self.image_pubs = {}
        self.camera_info_pubs = {}

        self.bridge = CvBridge()
        self.data_folder = '/data/sparse4dv3/bag_data'
        self.cameras = [
            {
                'name': 'CAM_FRONT_LEFT',
                'image_path': os.path.join(self.data_folder, 'CAM_FRONT_LEFT.png'),
                'camera_info_path': os.path.join(self.data_folder, 'CAM_FRONT_LEFT', 'camera_info.yaml'),
                'tf_data': {
                    'translation': [1.52387798135, 0.494631336551, 1.50932822144],
                    'rotation': [-0.6736266522251881, 0.21214015046209478, -0.21122827103904068, 0.6757265034669446]
                }
            },
            {
                'name': 'CAM_FRONT_RIGHT',
                'image_path': os.path.join(self.data_folder, 'CAM_FRONT_RIGHT.png'),
                'camera_info_path': os.path.join(self.data_folder, 'CAM_FRONT_RIGHT', 'camera_info.yaml'),
                'tf_data': {
                    'translation': [1.5508477543, -0.493404796419, 1.49574800619],
                    'rotation': [-0.2026940577919598, 0.6824507824531167, -0.6713610884174485, 0.2060347966337182]
                }
            }
            # Can Add more cameras if needed.
        ]
        for camera in self.cameras:
            camera_name = camera['name']
            self.image_pubs[camera_name] = self.create_publisher(
                Image, f'/{camera_name}/image_rect', 10)
            self.camera_info_pubs[camera_name] = self.create_publisher(
                CameraInfo, f'/{camera_name}/camera_info', 10)
            camera['image'] = cv2.imread(camera['image_path'])
            if camera['image'] is None:
                self.get_logger().error(f"Could not read image from {camera['image_path']}")
                continue
            with open(camera['camera_info_path'], 'r') as f:
                camera['camera_info'] = yaml.safe_load(f)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.publish_data)

    def publish_data(self):
        stamp = self.get_clock().now().to_msg()
        transforms = []

        for camera in self.cameras:
            camera_name = camera['name']
            ros_image = self.bridge.cv2_to_imgmsg(camera['image'], encoding='bgr8')
            ros_image.header.stamp = stamp
            self.image_pubs[camera_name].publish(ros_image)
            camera_info_msg = CameraInfo()
            camera_info_msg.header.stamp = stamp
            camera_info_msg.header.frame_id = camera_name  # Using camera name as frame_id.
            camera_info_msg.width = camera['camera_info']['width']
            camera_info_msg.height = camera['camera_info']['height']
            camera_info_msg.distortion_model = camera['camera_info']['distortion_model']
            camera_info_msg.d = camera['camera_info']['d']
            camera_info_msg.k = camera['camera_info']['k']
            camera_info_msg.r = camera['camera_info']['r']
            camera_info_msg.p = camera['camera_info']['p']
            camera_info_msg.binning_x = camera['camera_info']['binning_x']
            camera_info_msg.binning_y = camera['camera_info']['binning_y']
            roi = camera['camera_info']['roi']
            camera_info_msg.roi.x_offset = roi['x_offset']
            camera_info_msg.roi.y_offset = roi['y_offset']
            camera_info_msg.roi.height = roi['height']
            camera_info_msg.roi.width = roi['width']
            camera_info_msg.roi.do_rectify = roi['do_rectify']
            self.camera_info_pubs[camera_name].publish(camera_info_msg)

            # --- Prepare TF for the camera (base_link -> camera frame) ---
            tf_msg_camera = TransformStamped()
            tf_msg_camera.header.stamp = stamp
            tf_msg_camera.header.frame_id = 'base_link'
            tf_msg_camera.child_frame_id = camera_name
            tf_msg_camera.transform.translation.x = camera['tf_data']['translation'][0]
            tf_msg_camera.transform.translation.y = camera['tf_data']['translation'][1]
            tf_msg_camera.transform.translation.z = camera['tf_data']['translation'][2]
            tf_msg_camera.transform.rotation.x = camera['tf_data']['rotation'][0]
            tf_msg_camera.transform.rotation.y = camera['tf_data']['rotation'][1]
            tf_msg_camera.transform.rotation.z = camera['tf_data']['rotation'][2]
            tf_msg_camera.transform.rotation.w = camera['tf_data']['rotation'][3]
            transforms.append(tf_msg_camera)

        # --- Prepare Global TFs ---
        # TF: map -> base_link
        tf_msg_map_base = TransformStamped()
        tf_msg_map_base.header.stamp = stamp
        tf_msg_map_base.header.frame_id = 'map'
        tf_msg_map_base.child_frame_id = 'base_link'
        tf_msg_map_base.transform.translation.x = 397.8606615606245  # Adjust as needed.
        tf_msg_map_base.transform.translation.y = 1121.0773884177784
        tf_msg_map_base.transform.translation.z = 0.0
        tf_msg_map_base.transform.rotation.x = -0.01704046260824109
        tf_msg_map_base.transform.rotation.y = 0.012307579227643951
        tf_msg_map_base.transform.rotation.z = -0.561608772194064
        tf_msg_map_base.transform.rotation.w = 0.8271358613446328
        transforms.append(tf_msg_map_base)

        # TF: base_link -> LIDAR_TOP
        tf_msg_lidar = TransformStamped()
        tf_msg_lidar.header.stamp = stamp
        tf_msg_lidar.header.frame_id = 'base_link'
        tf_msg_lidar.child_frame_id = 'LIDAR_TOP'
        tf_msg_lidar.transform.translation.x = 0.943713  # Adjust as needed.
        tf_msg_lidar.transform.translation.y = 0.0
        tf_msg_lidar.transform.translation.z = 1.84023
        tf_msg_lidar.transform.rotation.x = -0.006492242056004365
        tf_msg_lidar.transform.rotation.y = 0.010646214713995808
        tf_msg_lidar.transform.rotation.z = -0.7063073142877817
        tf_msg_lidar.transform.rotation.w = 0.7077955119163518
        transforms.append(tf_msg_lidar)

        # TF: base_link -> LIDAR_TOP_GLOBAL
        tf_msg_lidar_map = TransformStamped()
        tf_msg_lidar_map.header.stamp = stamp
        tf_msg_lidar_map.header.frame_id = 'base_link'
        tf_msg_lidar_map.child_frame_id = 'LIDAR_TOP_GLOBAL'
        tf_msg_lidar_map.transform.translation.x = 397.8606615606245  # Adjust as needed.
        tf_msg_lidar_map.transform.translation.y = 1121.0773884177784
        tf_msg_lidar_map.transform.translation.z = 0.0
        tf_msg_lidar_map.transform.rotation.x = -0.01704046260824109
        tf_msg_lidar_map.transform.rotation.y = 0.012307579227643951
        tf_msg_lidar_map.transform.rotation.z = -0.561608772194064
        tf_msg_lidar_map.transform.rotation.w = 0.8271358613446328
        transforms.append(tf_msg_lidar_map)

        # TF: base_link -> CAM_FRONT_LEFT_GLOBAL
        tf_msg_camfl_map = TransformStamped()
        tf_msg_camfl_map.header.stamp = stamp
        tf_msg_camfl_map.header.frame_id = 'base_link'
        tf_msg_camfl_map.child_frame_id = 'CAM_FRONT_LEFT_GLOBAL'
        tf_msg_camfl_map.transform.translation.x = 397.7928370298025
        tf_msg_camfl_map.transform.translation.y = 1121.2515629175316
        tf_msg_camfl_map.transform.translation.z = 0.0
        tf_msg_camfl_map.transform.rotation.x = -0.01640712395621434
        tf_msg_camfl_map.transform.rotation.y = 0.012085767158384362
        tf_msg_camfl_map.transform.rotation.z = -0.5654416658900115
        tf_msg_camfl_map.transform.rotation.w = 0.824536514043622
        transforms.append(tf_msg_camfl_map)

        # TF: base_link -> CAM_FRONT_RIGHT_GLOBAL
        tf_msg_camfr_map = TransformStamped()
        tf_msg_camfr_map.header.stamp = stamp
        tf_msg_camfr_map.header.frame_id = 'base_link'
        tf_msg_camfr_map.child_frame_id = 'CAM_FRONT_RIGHT_GLOBAL'
        tf_msg_camfr_map.transform.translation.x = 397.816711742893
        tf_msg_camfr_map.transform.translation.y = 1121.1900306675905
        tf_msg_camfr_map.transform.translation.z = 0.0
        tf_msg_camfr_map.transform.rotation.x = -0.016597532223889603
        tf_msg_camfr_map.transform.rotation.y = 0.012173993697703037
        tf_msg_camfr_map.transform.rotation.z = -0.564100089123712
        tf_msg_camfr_map.transform.rotation.w = 0.8254498199479754
        transforms.append(tf_msg_camfr_map)

        # Publish all collected transforms on /tf.
        self.tf_broadcaster.sendTransform(transforms)

def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
