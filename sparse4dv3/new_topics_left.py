import os
import cv2
import copy
import sys
import time
import numpy as np
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

class RectifyImageNode(Node):
    def __init__(self):
        super().__init__('rectify_image_node')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, '/device/camera_front_center_left/image_8bit_color', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/device/camera_front_center_left/camera_info', self.camera_info_callback, 10)
        self.image_pub = self.create_publisher(
            Image, '/CAM_FRONT_LEFT/image_rect', 10)
        self.camera_info_pub = self.create_publisher(
            CameraInfo, '/CAM_FRONT_LEFT/camera_info', 10)
        
        self.K = None
        self.D = None
        self.map1 = None
        self.map2 = None
        self.image_size = None
        self.newK = None  

    def camera_info_callback(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)
        self.D = np.array(msg.d)
        self.image_size = (msg.width, msg.height)
        balance = 0.0  
        self.newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, self.image_size, np.eye(3), balance=balance) 
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.newK, self.image_size, cv2.CV_16SC2)
        rectified_camera_info = CameraInfo()
        rectified_camera_info.header = msg.header
        new_width, new_height = 1600, 900
        rectified_camera_info.width = new_width
        rectified_camera_info.height = new_height
        orig_width, orig_height = self.image_size
        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

        newK_resized = self.newK.copy()
        newK_resized[0, 0] *= scale_x  
        newK_resized[1, 1] *= scale_y  
        newK_resized[0, 2] *= scale_x 
        newK_resized[1, 2] *= scale_y  

        rectified_camera_info.k = list(newK_resized.flatten())
        rectified_camera_info.d = []
        rectified_camera_info.r = list(np.eye(3).flatten())
        P = np.zeros((3, 4))
        P[:3, :3] = newK_resized
        rectified_camera_info.p = list(P.flatten())
        rectified_camera_info.distortion_model = ""
        rectified_camera_info.header.stamp = msg.header.stamp
        rectified_camera_info.header.frame_id = msg.header.frame_id
        self.camera_info_pub.publish(rectified_camera_info)

    def image_callback(self, msg):
        if self.map1 is None or self.map2 is None:
            return
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        rectified = cv2.remap(cv_image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        resized = cv2.resize(rectified, (1600, 900), interpolation=cv2.INTER_LINEAR)
        rectified_msg = self.bridge.cv2_to_imgmsg(resized, encoding='rgb8')
        rectified_msg.header.stamp = msg.header.stamp
        rectified_msg.header.frame_id = msg.header.frame_id
        self.image_pub.publish(rectified_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RectifyImageNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
