import torch

from .deformable_aggregation import DeformableAggregationFunction, KeypointsGenerationFunction
from .deformable_aggregation import GroupWeightsGenerationFunction, KeypointsProjectionFunction
from .deformable_aggregation import SparseBox3DRefinementFunction
from .deformable_aggregation import GraphModelFunction

def deformable_aggregation_function(
    feature_maps,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
):
    return DeformableAggregationFunction.apply(
        feature_maps,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    )

def keypoints_generation_function(anchors, fixed_scale, learn_scale):
    return KeypointsGenerationFunction.apply(anchors, fixed_scale, learn_scale)

def group_weights_generation_function(instance_feature, anchor_embed, projection_mat, weights, biases, output_dims, num_groups):
    return GroupWeightsGenerationFunction.apply(instance_feature, anchor_embed, projection_mat, weights, biases, output_dims, num_groups)

def keypoints_projection_function(keypoints, projection_mat, image_wh):
    return KeypointsProjectionFunction.apply(keypoints, projection_mat, image_wh)

def sparse_box3d_refinement_function(instance_feature, anchor, anchor_embed, time_interval,
    weights, biases, scale, embed_dims, output_dims, num_cls, return_cls):
    return SparseBox3DRefinementFunction.apply(instance_feature, anchor, anchor_embed, time_interval,
        weights, biases, scale, embed_dims, output_dims, num_cls, return_cls)

def graph_model_function(query, key, value, query_pos, key_pos, weights, biases, num_heads, head_dim):
    return GraphModelFunction.apply(query, key, value, query_pos, key_pos, weights, biases, num_heads, head_dim)

def feature_maps_format(feature_maps, inverse=False):
    if inverse:
        col_feats, spatial_shape, scale_start_index = feature_maps
        num_cams, num_levels = spatial_shape.shape[:2]

        split_size = spatial_shape[..., 0] * spatial_shape[..., 1]
        split_size = split_size.cpu().numpy().tolist()

        idx = 0
        cam_split = [1]
        cam_split_size = [sum(split_size[0])]
        for i in range(num_cams - 1):
            if not torch.all(spatial_shape[i] == spatial_shape[i + 1]):
                cam_split.append(0)
                cam_split_size.append(0)
            cam_split[-1] += 1
            cam_split_size[-1] += sum(split_size[i + 1])
        mc_feat = [
            x.unflatten(1, (cam_split[i], -1))
            for i, x in enumerate(col_feats.split(cam_split_size, dim=1))
        ]

        spatial_shape = spatial_shape.cpu().numpy().tolist()
        mc_ms_feat = []
        shape_index = 0
        for i, feat in enumerate(mc_feat):
            feat = list(feat.split(split_size[shape_index], dim=2))
            for j, f in enumerate(feat):
                feat[j] = f.unflatten(2, spatial_shape[shape_index][j])
                feat[j] = feat[j].permute(0, 1, 4, 2, 3)
            mc_ms_feat.append(feat)
            shape_index += cam_split[i]
        return mc_ms_feat

    if isinstance(feature_maps[0], (list, tuple)):
        formated = [feature_maps_format(x) for x in feature_maps]
        col_feats = torch.cat([x[0] for x in formated], dim=1)
        spatial_shape = torch.cat([x[1] for x in formated], dim=0)
        scale_start_index = torch.cat([x[2] for x in formated], dim=0)
        return [col_feats, spatial_shape, scale_start_index]

    bs, num_cams = feature_maps[0].shape[:2]
    spatial_shape = []

    col_feats = []
    for i, feat in enumerate(feature_maps):
        spatial_shape.append(feat.shape[-2:])
        col_feats.append(
            torch.reshape(feat, (bs, num_cams, feat.shape[2], -1))
        )

    col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
    spatial_shape = [spatial_shape] * num_cams
    spatial_shape = torch.tensor(
        spatial_shape,
        dtype=torch.int64,
        device=col_feats.device,
    )
    scale_start_index = spatial_shape[..., 0] * spatial_shape[..., 1]
    scale_start_index = scale_start_index.flatten().cumsum(dim=0)
    scale_start_index = torch.cat(
        [torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]]
    )
    scale_start_index = scale_start_index.reshape(num_cams, -1)

    feature_maps = [
        col_feats,
        spatial_shape,
        scale_start_index,
    ]
    return feature_maps
