# Copyright (c) Horizon Robotics. All rights reserved.
from inspect import signature
import torch
import time

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .grid_mask import GridMask
from torch.profiler import record_function

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["Sparse4D"]

def perf_counter_sync():
    torch.cuda.synchronize()
    return time.perf_counter()

@DETECTORS.register_module()
class Sparse4D(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
        
    ):
        super(Sparse4D, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            with record_function("GridMask"):
                img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            with record_function("Backbone"):
                feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            with record_function("Backbone"):
                feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            with record_function("Neck"):
                feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            with record_function("DepthBranch"):
                depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            with record_function("PreprocessDFA"):
                feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)
    
    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img)
        with record_function("SparseHead"):
            model_outs = self.head(feature_maps, data)
        with record_function("PostProcessSparseBox3DDecoder"):
            results = self.head.post_process(model_outs)
        output = [{"img_bbox": result} for result in results]
        return output

    @force_fp32(apply_to=("img",))
    def simple_test_with_profile(self, img, **data):
        ts = []
        ts.append(perf_counter_sync())
        feature_maps = self.extract_feat(img)
        ts.append(perf_counter_sync())
        with record_function("SparseHead"):
            model_outs = self.head(feature_maps, data)
        ts.append(perf_counter_sync())
        with record_function("PostProcessSparseBox3DDecoder"):
            results = self.head.post_process(model_outs)
        output = [{"img_bbox": result} for result in results]
        ts.append(perf_counter_sync())

        profile = dict()
        profile['feat']  = 1000 * (ts[1] - ts[0])
        profile['head']  = 1000 * (ts[2] - ts[1])
        profile['post']  = 1000 * (ts[3] - ts[2])
        profile['total'] = 1000 * (ts[3] - ts[0])

        return output, profile

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
