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
        super().__init__('rectify_image_node1')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, '/CAM_FRONT_RIGHT/image_8bit_color', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/CAM_FRONT_RIGHT/camera_info', self.camera_info_callback, 10)
        self.image_pub = self.create_publisher(
            Image, '/CAM_FRONT_RIGHT/image_rect', 10)
        self.K = None
        self.D = None
        self.map1 = None
        self.map2 = None
        self.image_size = None

    def camera_info_callback(self, msg):

        self.K = np.array(msg.k).reshape(3, 3)
        self.D = np.array(msg.d)
        self.image_size = (msg.width, msg.height)
        balance = 0.0 
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, self.image_size, np.eye(3), balance=balance) 
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), newK, self.image_size, cv2.CV_16SC2)

    def image_callback(self, msg):
        if self.map1 is None or self.map2 is None:
            return  
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        rectified = cv2.remap(cv_image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        resized = cv2.resize(rectified, (1600, 900), interpolation=cv2.INTER_LINEAR) #resize

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
