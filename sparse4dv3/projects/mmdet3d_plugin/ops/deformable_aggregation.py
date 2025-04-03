import torch
from torch.autograd.function import Function, once_differentiable

from . import deformable_aggregation_ext


class DeformableAggregationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        # output: [bs, num_pts, num_embeds]
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()
        output = deformable_aggregation_ext.deformable_aggregation_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        deformable_aggregation_ext.deformable_aggregation_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
        )

class KeypointsGenerationFunction(Function):
    @staticmethod
    def forward(ctx, anchors, fixed_scale, learn_scale):
        output = deformable_aggregation_ext.keypoints_generation(anchors, fixed_scale, learn_scale)
        return output

class GroupWeightsGenerationFunction(Function):
    @staticmethod
    def forward(ctx, instance_feature, anchor_embed, projection_mat, weights, biases, output_dims, num_groups):
        output = deformable_aggregation_ext.group_weights_generation(
            instance_feature, anchor_embed, projection_mat, weights, biases, output_dims, num_groups)
        return output

class KeypointsProjectionFunction(Function):
    @staticmethod
    def forward(ctx, keypoints, projection_mat, image_wh):
        output = deformable_aggregation_ext.keypoints_projection(keypoints, projection_mat, image_wh)
        return output

class SparseBox3DRefinementFunction(Function):
    @staticmethod
    def forward(ctx, instance_feature, anchor, anchor_embed, time_interval, weights, biases, scale, embed_dims, output_dims, num_cls, return_cls):
        output = deformable_aggregation_ext.sparse_box3d_refinement(
            instance_feature, anchor, anchor_embed, time_interval, weights, biases, scale, embed_dims, output_dims, num_cls, return_cls)
        return output

class GraphModelFunction(Function):
    @staticmethod
    def forward(ctx, query, key, value, query_pos, key_pos, weights, biases, num_heads, head_dim):
        if key is None: key = torch.empty(0)
        if value is None: value = torch.empty(0)
        if key_pos is None: key_pos = torch.empty(0)
        output = deformable_aggregation_ext.graph_model(query, key, value, query_pos, key_pos, weights, biases, num_heads, head_dim)
        return output
