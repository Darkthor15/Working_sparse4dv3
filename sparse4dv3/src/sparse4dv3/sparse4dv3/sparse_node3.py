# Python Libraries
import os
import cv2
import torch
import copy
import sys
sys.path.insert(0, '/data/sparse4dv3')
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import pyquaternion
import transforms3d.quaternions as tq
from mmcv import Config
from mmcv.runner import wrap_fp16_model, load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import DataContainer as DC
from torch2trt import TRTModule
# ROS2 Libraries
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer, SimpleFilter 
from sparse_msgs.msg import CustomTFMessage, BoxInfo, BBoxes3D
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from projects.mmdet3d_plugin.datasets.utils import draw_lidar_bbox3d


class TFMergeFilter(SimpleFilter):
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger
        self.stored_odom = None       
        self.stored_sensor = None     

    def add(self, msg):
        if msg.header.frame_id == "odom":
            if self.stored_sensor is not None:

                merged_msg = self.merge_msgs(odom_msg=msg, sensor_msg=self.stored_sensor, latter=msg)
                self.stored_sensor = None
                self.stored_odom = None
                if self.logger:
                    transforms = [t.child_frame_id for t in merged_msg.tf_message.transforms]
                    self.logger.info(
                        f"stamp: {merged_msg.header.stamp.sec}.{merged_msg.header.stamp.nanosec}, "
                        f"transforms: {transforms}"
                    )
                self.signalMessage(merged_msg)
            else:

                self.stored_odom = msg
                if self.logger:
                    self.logger.info("Stored odom message.")
        elif msg.header.frame_id == "base_link":

            if self.stored_odom is not None:
                merged_msg = self.merge_msgs(odom_msg=self.stored_odom, sensor_msg=msg, latter=msg)
                self.stored_odom = None
                self.stored_sensor = None
                if self.logger:
                    transforms = [t.child_frame_id for t in merged_msg.tf_message.transforms]
                    self.logger.info(
                        f"stamp: {merged_msg.header.stamp.sec}.{merged_msg.header.stamp.nanosec}, "
                        f"transforms: {transforms}"
                    )
                self.signalMessage(merged_msg)
            else:

                self.stored_sensor = msg
                if self.logger:
                    self.logger.info("Stored base_link message.")
        else:

            if self.logger:
                self.logger.info(f"Received message with unknown frame_id '{msg.header.frame_id}'")
            self.signalMessage(msg)

    def merge_msgs(self, odom_msg, sensor_msg, latter):
        merged_msg = CustomTFMessage()
        merged_msg.header = latter.header
        merged_msg.header.frame_id = "odom"
        merged_transforms = []
        if odom_msg.tf_message.transforms:
            merged_transforms.extend(odom_msg.tf_message.transforms)
        if sensor_msg.tf_message.transforms:
            merged_transforms.extend(sensor_msg.tf_message.transforms)
        merged_msg.tf_message.transforms = merged_transforms
        return merged_msg


def import_plugins(cfg:Config) -> None:
    """Import custom plugins.
    
    Args:
        cfg (Config): Configs loaded from config file.
    """
    if cfg.plugin:
        import importlib
        if hasattr(cfg, "plugin_dir"):
            sys.path.append(os.getcwd())
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split("/")
            _module_path = _module_dir[0]

            for m in _module_dir[1:]:
                _module_path = _module_path + "." + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)


def create_model(cfg: Config, ckpt:str) -> torch.nn.Module:
    """Create Sparse4Dv3 Model from checkpoint.

    Args:
        cfg (Config): Configs loaded from config file.
        ckpt (str): Path to PyTorch checkpoint(.pth) for model.
    
    Returns:
        model (nn.Module): Sparse4Dv3 Pytorch model.
    """
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get('fp16', None):
            wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, ckpt, map_location="cpu")
    if cfg.use_tensorrt:
        for k, v in cfg.trt_paths.items():
            if k == 'backbone' and v is not None:
                model.img_backbone = TRTModule()
                model.img_backbone.load_state_dict(torch.load(v))
            elif k == 'neck' and v is not None:
                model.img_neck = TRTModule()
                model.img_neck.load_state_dict(torch.load(v))
            elif k == 'encoder' and v is not None:
                model.head.anchor_encoder = TRTModule()
                model.head.anchor_encoder.load_state_dict(torch.load(v))
            elif k == 'temp_encoder' and v is not None:
                model.head.temp_anchor_encoder = TRTModule()
                model.head.temp_anchor_encoder.load_state_dict(torch.load(v))

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    # palette for visualization in segmentation tasks
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    return model


def obtain_sensor2lidar_rt(
    l2e_t:list, l2e_r:list, e2g_t:list, e2g_r:list,
    s2e_t:list, s2e_r:list, se2g_t:list, se2g_r:list
) -> tuple:
    """Obtain the info with RT from sensor to LiDAR.
    
    Args:
        l2e_t (list): Translation from LiDAR to ego in (x, y, z).
        l2e_r (list): Rotation quat from LiDAR to ego in (w, x, y, z).
        e2g_t (list): Translation from ego to global in (x, y, z).
        e2g_r (list): Rotation quat from ego to global in (w, x, y, z).
        s2e_t (list): Translation from sensor to ego in (x, y, z).
        s2e_r (list): Rotation quat from sensor to ego in (w, x, y, z).
        se2g_t (list): Translation from sensor ego to global in (x, y, z).
        se2g_r (list): Rotation quat from sensor ego to global in (w, x, y, z).
    
    Returns:
        s2l_r (np.ndarray): Sensor to LiDAR rotation matrix.
        s2l_t (np.ndarray): Sensor to LiDAR translation vector.
    """
    # sensor->ego->global->ego'->lidar
    l2e_r_mat = pyquaternion.Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = pyquaternion.Quaternion(e2g_r).rotation_matrix
    s2e_r_mat = pyquaternion.Quaternion(s2e_r).rotation_matrix
    se2g_r_mat = pyquaternion.Quaternion(se2g_r).rotation_matrix

    R = (s2e_r_mat.T @ se2g_r_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (np.array(s2e_t) @ se2g_r_mat.T + np.array(se2g_t)) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        np.array(e2g_t) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + np.array(l2e_t) @ np.linalg.inv(l2e_r_mat).T
    )
    s2l_r = R.T  # points @ R.T + T
    s2l_t = T

    return s2l_r, s2l_t


def get_augmentation(cfg:Config) -> dict:
    """Get Image Augmentation parameters.
    
    Args:
        cfg (Config): Configs loaded from config file.
    
    Returns:
        aug_config (Dict): Dictionary with augmentation parameters.
    """
    if cfg.data_aug_conf is None:
        return None
    H, W = cfg.data_aug_conf["H"], cfg.data_aug_conf["W"]
    fH, fW = cfg.data_aug_conf["final_dim"]
    resize = max(fH / H, fW / W)
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = (
        int((1 - np.mean(cfg.data_aug_conf["bot_pct_lim"])) * newH)
        - fH
    )
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    flip = False
    rotate = 0
    rotate_3d = 0
    aug_config = {
        "resize": resize,
        "resize_dims": resize_dims,
        "crop": crop,
        "flip": flip,
        "rotate": rotate,
        "rotate_3d": rotate_3d,
    }
    return aug_config

def matrix(translation, rotation):
    T = np.eye(4)
    T[:3, :3] = pyquaternion.Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T

def process_transforms(infos:list, tfs:list, cfg:Config) -> dict:
    """ Obtain lidar2global and lidar2image(if any) transforms in Dictionary format.

    Args:
        infos (list): Camera intrinsic information for all cameras.
        tfs (list): List of transforms from /tf topic.
        cfg (Config): Configs loaded from config file.
    
    Returns:
        input_dict (Dict): Dictionary containing tranforms for lidar and cameras.
    """
    timestamp = tfs[0].header.stamp.sec + (tfs[0].header.stamp.nanosec / 1e9)
    lidar2ego_translation = None
    lidar2ego_rotation = None
    ego2global_translation = None
    ego2global_rotation = None

    # Get LiDAR transforms first
    # (Needed for sensor2lidar_RT calculation)
    for tf in tfs:
        if tf.child_frame_id == "LIDAR_TOP":  #LIDAR_TOP
            lidar2ego_translation = [
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z
            ]
            lidar2ego_rotation = [
                tf.transform.rotation.w,
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z
            ]
        elif tf.child_frame_id == "base_link":
            ego2global_translation = [
                tf.transform.translation.x,
                tf.transform.translation.y,
                0
                # tf.transform.translation.z
            ]
            ego2global_rotation = [
                tf.transform.rotation.w,
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z
            ]
    # lidar2ego_translation = [0.943713, 0.0, 1.84023]
    # lidar2ego_rotation = [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817]
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = pyquaternion.Quaternion(
        lidar2ego_rotation).rotation_matrix
    lidar2ego[:3, 3] = np.array(lidar2ego_translation)
    ego2global = np.eye(4)
    ego2global[:3, :3] = pyquaternion.Quaternion(
        ego2global_rotation).rotation_matrix
    ego2global[:3, 3] = np.array(ego2global_translation)
    lidar2global = ego2global @ lidar2ego

    input_dict = dict(
        timestamp=timestamp,
        lidar2ego_translation=lidar2ego_translation,
        lidar2ego_rotation=lidar2ego_rotation,
        ego2global_translation=ego2global_translation,
        ego2global_rotation=ego2global_rotation,
        lidar2global=lidar2global,
    )

    # Get camera transforms next
    if cfg.input_modality["use_camera"]:
        cams = {k:v for v, k in enumerate(cfg.cams)}
        sensor2ego_translation = [None] * len(cams)
        sensor2ego_rotation = [None] * len(cams)
        sensor_ego2global_translation = [None] * len(cams)
        sensor_ego2global_rotation = [None] * len(cams)
        # new_sensor_ego2global_translation= [None]* len(cams)
        # new_sensor_ego2global_rotation = [None]* len(cams)
        lidar2img_rts = [None] * len(cams)
        cam_intrinsic = [None] * len(cams)
        for tf in tfs:
            for cam_name, frame_id in cfg.child_frame_mapping.items():
                if tf.child_frame_id == frame_id:
                    sensor2ego_translation[cams[cam_name]] = [
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        tf.transform.translation.z
                    ]
                    sensor2ego_rotation[cams[cam_name]] = [
                        tf.transform.rotation.w,
                        tf.transform.rotation.x,
                        tf.transform.rotation.y,
                        tf.transform.rotation.z
                    ]
                elif tf.child_frame_id[:-7] == frame_id:  #tf.child_frame_id[:-7] == frame_id which is wrong tf.child_frame_id == "base_link"
                    sensor_ego2global_translation[cams[cam_name]] = [
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        tf.transform.translation.z
                    ]
                    sensor_ego2global_rotation[cams[cam_name]] = [
                        tf.transform.rotation.w,
                        tf.transform.rotation.x,
                        tf.transform.rotation.y,
                        tf.transform.rotation.z
                    ]

        # for tf in tfs:
        #     for i in range(len(cams)):
        #         if tf.child_frame_id == "base_link":

        #             sensor_ego2global_translation[i] = [
        #                 tf.transform.translation.x,
        #                 tf.transform.translation.y,
        #                 tf.transform.translation.z
        #         ]
        #             sensor_ego2global_rotation[i] = [
        #                 tf.transform.rotation.w,
        #                 tf.transform.rotation.x,
        #                 tf.transform.rotation.y,
        #                 tf.transform.rotation.z
        #         ]
        # lidar2ego_rotation = []
        # lidar2ego_translation = []
        # sensor2ego_translation = []
        
        # sensor_ego2global_translation[0] = ego2global_translation + np.array([0, 0.32, 0])
        # sensor_ego2global_translation[1] = ego2global_translation + np.array([0, -0.32, 0])
        # sensor_ego2global_rotation[0] = ego2global_rotation 
        # sensor_ego2global_rotation[1] = ego2global_rotation 


        sensor2ego = np.eye(4)
        ego2global = np.eye(4)
        # for i in range(len(cams)):
        #     sensor2ego[:3, :3] = pyquaternion.Quaternion(
        #     sensor2ego_rotation[i]).rotation_matrix
        #     sensor2ego[:3, 3] = np.array(sensor2ego_translation[i])

        #     ego2global[:3, :3] = pyquaternion.Quaternion(s
        #     ego2global_rotation[i]).rotation_matrix
        #     ego2global[:3, 3] = np.array(ego2global_translation[i])
        #     sensor2global = ego2global @ sensor2ego
        #     sensor_ego2global_translation[i] = sensor2global[:3, 3].tolist()
        #     q = pyquaternion.Quaternion(matrix= sensor2global[:3, :3])
        #     sensor_ego2global_rotation[i] = [q.w, q.x, q.y, q.z]

        # T_global_ego = matrix(ego2global_translation, ego2global_rotation)
        # for i in range(len(cams)):
        #     if sensor2ego_translation[i] is not None and sensor2ego_rotation[i] is not None:
        #         T_sensor2ego = matrix(sensor2ego_translation[i], sensor2ego_rotation[i])
        #         T_global_sensor = T_global_ego @ T_sensor2ego
        #         new_sensor_ego2global_translation[i] = T_global_sensor[:3, 3].tolist()
        #         q = pyquaternion.Quaternion(matrix=T_global_sensor[:3, :3])
        #         new_sensor_ego2global_rotation[i] = [q.w, q.x, q.y, q.z]


        # Get LiDAR to Image projection matrices for all cameras
        for i in range(len(cams)):
            sensor2lidar_rotation, sensor2lidar_translation = obtain_sensor2lidar_rt(
                lidar2ego_translation, lidar2ego_rotation,
                ego2global_translation, ego2global_rotation,
                sensor2ego_translation[i], sensor2ego_rotation[i],
                sensor_ego2global_translation[i], sensor_ego2global_rotation[i]
            )
            lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
            lidar2cam_t = (sensor2lidar_translation @ lidar2cam_r.T)
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t  # lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = copy.deepcopy(infos[i])
            cam_intrinsic[i] = intrinsic
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = viewpad @ lidar2cam_rt.T
            lidar2img_rts[i] = lidar2img_rt
        input_dict.update(
            dict(
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsic,
                cam_intrinsicintrinsic = cam_intrinsic,
                viewpad = viewpad,
                # lidar2global=lidar2global,
                # ego2global_translation=ego2global_translation,
                # ego2global_rotation=ego2global_rotation,
                # lidar2img_rt = lidar2img_rt
                # sensor_ego2global_translation = sensor_ego2global_translation,
                # sensor_ego2global_rotation = sensor_ego2global_rotation,
                # lidar2ego_rotation = lidar2ego_rotation,
                # lidar2ego_translation = lidar2ego_translation
            )
        )
    return input_dict


class Sparse4Dv3Node(Node):

    def __init__(self):
        super().__init__("sparse4dv3_node")

        # Configs and Necessary variables
        self.get_logger().info("Initializing...")
        self.cfg = Config.fromfile('projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py')
        self.cams = self.cfg.get("cams", None)
        self.images = [None] * len(self.cams)
        self.cam_intrinsics = [None] * len(self.cams)
        self.cam_wh = [None] * len(self.cams)
        self.tf_info = None
        self.count = 0
        self.outs = []
        # Detection Model and Preprocess Pipeline
        if hasattr(self.cfg, 'plugin'):
            import_plugins(self.cfg)
        self.model = create_model(self.cfg, 'ckpt/sparse4dv3_r50.pth')
        self.model.cuda().eval()
        self.aug_config = get_augmentation(self.cfg)
        self.pipeline = Compose(self.cfg.test_pipeline)
        self.message_time_threshold = 30
        self.visualize = True
        self.pub_viz_image = self.create_publisher(Image, '/viz_image', 10)
        
        # Subscribers
        self.img_subs = [
            Subscriber(
                self,
#                CompressedImage,
                Image,
#                f'/{cam}/image_rect_compressed',
                f'/{cam}/image_rect',
            ) for cam in self.cams
        ]
        self.cam_info_subs = [
            Subscriber(
                self,
                CameraInfo,
                f'/{cam}/camera_info',
            ) for cam in self.cams
        ]
        self.tf_subs = Subscriber(
            self,
            CustomTFMessage,
            '/tf_stamped',
        )

        # self.tf_merge_filter = TFMergeFilter(logger=self.get_logger())
        # self.tf_sub = Subscriber(self, CustomTFMessage, '/tf_stamped')
        # self.tf_sub.registerCallback(self.tf_merge_filter.add)

        all_subscribers = self.img_subs + self.cam_info_subs + [self.tf_subs]   #[self.tf_merge_filter]    #[self.tf_subs]
        
        # Time Synchronizer
        self.time_sync = ApproximateTimeSynchronizer(all_subscribers,
                                                    queue_size=10, slop=0.1)
        self.time_sync.registerCallback(self.sync_callback)

        self.bridge = CvBridge()

        # Publishers
        self.pub_boxes = self.create_publisher(
            BBoxes3D,
            '/bboxes3d',
            10,
        )
        self.pub_viz = self.create_publisher(MarkerArray, '/bbox_markers', 10)

        self.message_time = time.time()
        self.dest_timer = self.create_timer(1.0, self.check_timeout)
        self.get_logger().info("Node Ready.")
        
    # Callbacks
    def sync_callback(self, *msg):
        img_msgs = msg[:len(self.cams)]
        cam_info_msgs = msg[len(self.cams):-1]
        tf_msgs = msg[-1]
        self.get_logger().info("Received all info")
        self.message_time = time.time()

        for i in range(len(self.cams)):
            #img_decode = cv2.imdecode(np.frombuffer(img_msgs[i].data, np.uint8),
            #                            cv2.IMREAD_COLOR)
            #self.images[i] = img_decode
            self.images[i] = self.bridge.imgmsg_to_cv2(img_msgs[i], desired_encoding='bgr8')
            self.cam_intrinsics[i] = cam_info_msgs[i].k.reshape(3, 3)
            self.cam_wh[i] = [cam_info_msgs[i].width, cam_info_msgs[i].height]
        self.tf_info = tf_msgs.tf_message.transforms

        self.forward()
    
    def check_timeout(self):
        curr_time = time.time()
        time_since_last = curr_time - self.message_time
        if time_since_last > self.message_time_threshold:
            self.get_logger().info(f"No messages from topics for {self.message_time_threshold}[sec].")
            self.destroy_node()


    def publish_visualization(self, vis_img):
        """
        Converts the visualized image (a NumPy array) to a ROS Image message
        and publishes it on the /viz_image topic.
        """
        img_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding='rgb8')
        self.pub_viz_image.publish(img_msg)
        self.get_logger().info("Published updated visualization image on /viz_image.")

    def visualize_anchors(self, input_dict, out):

        vis_images = []
        img_tensors = []
        threshold = 0.35
        detection_boxes = []
        anchor_boxes = []
        
        img_norm_cfg = self.cfg.img_norm_cfg
        img_norm_mean = np.array(img_norm_cfg["mean"], dtype=np.float32)
        img_norm_std = np.array(img_norm_cfg["std"], dtype=np.float32)
        
        num_preds = len(out[0]["img_bbox"]["labels_3d"])
        for i in range(num_preds):
            score = float(out[0]["img_bbox"]["cls_scores"][i].cpu())
            # if score < threshold:
            #     continue
            pred_box = out[0]["img_bbox"]["boxes_3d"][i].cpu().unsqueeze(0)
            detection_boxes.append(pred_box)
            # anchor_box = self.model.head.decoder.decode_box(self.model.head.instance_bank.anchor).cpu()
            # anchor_boxes.append(anchor_box.unsqueeze(0))
        # self.get_logger().info(f" predbbox is {type(self.images)})")
        # self.get_logger().info(f" anchorbox is {anchor_boxes})")
        raw_imgs = input_dict["img"].squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
        raw_imgs = raw_imgs * img_norm_std + img_norm_mean

        num_det = len(detection_boxes)
        self.get_logger().info(f" bbox num_detection is {num_det}")
        if num_det == 0:
            self.get_logger().warn("No detections above threshold for visualization.")
            return
        boxes_to_draw = torch.cat(detection_boxes, dim = 0) #+ anchor_boxes
        colors = [(0, 255, 0)] * num_det #+ [(255, 0, 0)] * num_det
        # self.get_logger().info(f" raw_imgs is {raw_imgs})")
        # self.get_logger().info(f" boxes_to_draw is {boxes_to_draw})")
        # self.get_logger().info(f" proj_mat is {input_dict['projection_mat'][0]})")
        # Use the projection matrix of the first camera (adjust if you want multi-camera visualization)
        proj_mat = input_dict["projection_mat"][0]

        vis_img = draw_lidar_bbox3d(boxes_to_draw, raw_imgs, proj_mat, colors)
        # proj_mat = input_dict["lidar2img"]
        # anchors_sampled = anchor[::5]
        # vis_img = draw_lidar_bbox3d(anchors_sampled, raw_imgs, proj_mat)
        vis_images.append(vis_img)
        
        if vis_images:
            combined_img = np.hstack(vis_images)
            self.publish_visualization(combined_img)
        else:
            self.get_logger().warn("No images available for visualization.")

    
    # Forward to Sparse4Dv3
    def forward(self):
        
        if all(img is not None for img in self.images) \
        and all(cam_info is not None for cam_info in self.cam_intrinsics) \
        and (self.tf_info is not None):
            self.count += 1         
            boxes_3d = BBoxes3D()
            marker_array = MarkerArray()
            bboxes_viz = []
            self.get_logger().info(f"Preprocessing frame {self.count}...")
            input_dict = process_transforms(self.cam_intrinsics, self.tf_info, self.cfg)
            # dim = self.images[0].shape
            # self.images = 2 * [np.zeros(dim)]
            # self.get_logger().info(f" images {self.images[0][:100]}...")
            input_dict["img"] = self.images
            input_dict["aug_config"] = self.aug_config
            # if self.visualize:
            #     self.visualize_anchors(input_dict)  # for anchors visualization

            # self.get_logger().info(f"lidar2global is {input_dict['lidar2global']}")
            # self.get_logger().info(f"ego2global_translation is {input_dict['ego2global_translation']}")  
            # self.get_logger().info(f"ego2global_rotation is {input_dict['ego2global_rotation']}")
            # self.get_logger().info(f"sensor_ego2global_translation is {input_dict['sensor_ego2global_translation']}")  
            # self.get_logger().info(f"cam_intrinsic is {input_dict['cam_intrinsic']}")
            # self.get_logger().info(f"viewpad is  {input_dict['viewpad']}")
            input_dict = self.pipeline(input_dict)
            for k, v in input_dict.items():
                if isinstance(v, DC):
                    if k == 'img':
                        input_dict[k] = v.data.unsqueeze(dim=0).cuda()
                    elif k == 'img_metas':
                        input_dict[k] = [v.data]
                        ts = str(input_dict[k][0]["timestamp"]).split(".")
                        boxes_3d.header.stamp.sec = int(ts[0])
                        boxes_3d.header.stamp.nanosec = int(ts[1])
                elif isinstance(v, np.ndarray):
                    input_dict[k] = torch.from_numpy(v).unsqueeze(dim=0).cuda()
                else:
                    input_dict[k] = torch.tensor([v], dtype=torch.float64).cuda()
            self.get_logger().info("Preprocessing done. Inferencing...")
            with torch.no_grad():
                out = self.model(return_loss=False, rescale=True, **input_dict)
                self.outs.append(out[0])
            if self.visualize:
                self.visualize_anchors(input_dict, out)
            for i in range(len(out[0]["img_bbox"]["labels_3d"])):
                box = BoxInfo()
                box.id = int(i)
                box.bbox = out[0]["img_bbox"]["boxes_3d"][i].cpu().flatten().tolist()
                box.score = float(out[0]["img_bbox"]["cls_scores"][i].cpu())
                # if box.score < 0.35:
                #  continue
                box.label = int(out[0]["img_bbox"]["labels_3d"][i].cpu())
                box.instance = int(out[0]["img_bbox"]["instance_ids"][i].cpu())
                boxes_3d.boxes3d.append(box)

                marker = Marker()
                marker.header.frame_id = "LIDAR_TOP"
                marker.header.stamp = boxes_3d.header.stamp
                marker.ns = "bboxes"
                marker.id = int(i)
                marker.type = Marker.LINE_LIST
                marker.action = Marker.ADD
                marker.scale.x = 0.05  # Line Weight
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0

                marker.pose.orientation.w = 1.0  # no orientation

                # form mmdet3d_plugin/core/box3d.py
                # X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))  # undecoded
                x, y, z, w, l, h, sin_yaw, cos_yaw = box.bbox[:8]

                # Rotate the 8 vertices of a rectangular prism by the Yaw angle
                half_w, half_l, half_h = w / 2, l / 2, h / 2
                vertices = np.array([
                    [ half_w,  half_l,  half_h], [ half_w, -half_l,  half_h],
                    [-half_w, -half_l,  half_h], [-half_w,  half_l,  half_h],
                    [ half_w,  half_l, -half_h], [ half_w, -half_l, -half_h],
                    [-half_w, -half_l, -half_h], [-half_w,  half_l, -half_h]
                ])

                yaw_angle = box.bbox[6]
                yaw_rotation = R.from_euler('z', yaw_angle).as_matrix()

                # Rotate each vertex (transform around center)
                rotated_vertices = np.dot(vertices, yaw_rotation.T) + np.array([x, y, z])

                # Define the bounding box edges
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
                    (4, 5), (5, 6), (6, 7), (7, 4),  # Top
                    (0, 4), (1, 5), (2, 6), (3, 7)   # Side
                ]

                for start, end in edges:
                    p1 = Point(x=rotated_vertices[start][0], y=rotated_vertices[start][1], z=rotated_vertices[start][2])
                    p2 = Point(x=rotated_vertices[end][0], y=rotated_vertices[end][1], z=rotated_vertices[end][2])
                    marker.points.append(p1)
                    marker.points.append(p2)
                marker_array.markers.append(marker)
            self.get_logger().info("Inferencing done. Publishing topics...")
            # Publish output
            self.pub_boxes.publish(boxes_3d)
            self.get_logger().info("Published successfully!")
            # Publish markers
            self.pub_viz.publish(marker_array)
            self.get_logger().info("Published BBox markers.")
            # Reset
            self.images = [None] * len(self.cams)
            self.cam_intrmapo = None
    
    def print_eval_output(self, metrics_summary):
        self.get_logger().info('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            self.get_logger().info('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        self.get_logger().info('NDS: %.4f' % (metrics_summary['nd_score']))
        self.get_logger().info('Eval time: %.1fs' % metrics_summary['eval_time'])

        # self.get_logger().info per-class metrics.
        self.get_logger().info('')
        self.get_logger().info('Per-class results:')
        self.get_logger().info('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            self.get_logger().info('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err'],
                     class_tps[class_name]['attr_err']))

    def destroy_node(self):
        if self.cfg.ros_eval:
            self.get_logger().info("Evaluating outputs...")
            dataset = build_dataset(self.cfg.data.test)
            eval_kwargs = self.cfg.get("evaluation", {}).copy()
            eval_kwargs.pop("interval", None)
            eval_kwargs.update(dict(metric='bbox'))
            eval_result, metrics = dataset.evaluate(self.outs, **eval_kwargs)
            self.print_eval_output(metrics)
        self.get_logger().info("Exiting.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    sparse_node = Sparse4Dv3Node()
    rclpy.spin(sparse_node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()