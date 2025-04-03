
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <THC/THCAtomics.cuh>

#include <iostream>
#include <stdlib.h>

#define USE_OPTIMIZED_DFA

static const int WARP_SIZE = 32;
static constexpr int NUM_WARPS_PER_BLOCK = 4;

static int div_up(int total, int grain)
{
  return (total + grain - 1) / grain;
}

__device__ float bilinear_sampling(
    const float *&bottom_data, const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr
) {
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride = num_embeds;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

__device__ __inline__ int floorToInt(float x)
{
  return static_cast<int>(floor(x));
}

__device__ __inline__ float lerp(float v0, float v1, float t)
{
  return fmaf(t, v1, fmaf(-t, v0, v0));
}

__device__ __inline__ float bilinear_sampling_opt(const float *bottom_data, int h, int w, int num_embeds, float v_im, float u_im, int base_ptr)
{
  const int v_lo = floorToInt(v_im);
  const int u_lo = floorToInt(u_im);
  const int v_hi = v_lo + 1;
  const int u_hi = u_lo + 1;

  const float dv = v_im - v_lo;
  const float du = u_im - u_lo;

  const int u_stride = num_embeds;
  const int v_stride = w * u_stride;
  const int v_lo_ptr_offset = v_lo * v_stride;
  const int v_hi_ptr_offset = v_lo_ptr_offset + v_stride;
  const int u_lo_ptr_offset = u_lo * u_stride;
  const int u_hi_ptr_offset = u_lo_ptr_offset + u_stride;

  const int ptr1 = v_lo_ptr_offset + u_lo_ptr_offset + base_ptr; // v00
  const int ptr2 = v_lo_ptr_offset + u_hi_ptr_offset + base_ptr; // v01
  const int ptr3 = v_hi_ptr_offset + u_lo_ptr_offset + base_ptr; // v10
  const int ptr4 = v_hi_ptr_offset + u_hi_ptr_offset + base_ptr; // v11

  const float v1 = v_lo >= 0 && u_lo >= 0 ? bottom_data[ptr1] : 0;
  const float v2 = v_lo >= 0 && u_hi < w ? bottom_data[ptr2] : 0;
  const float v3 = v_hi < h && u_lo >= 0 ? bottom_data[ptr3] : 0;
  const float v4 = v_hi < h && u_hi < w ? bottom_data[ptr4] : 0;

  const float tmp0 = lerp(v1, v2, du);
  const float tmp1 = lerp(v3, v4, du);
  const float val = lerp(tmp0, tmp1, dv);

  return val;
}

__device__ void bilinear_sampling_grad(
    const float *&bottom_data, const float &weight,
    const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr,
    const float &grad_output,
    float *&grad_mc_ms_feat, float *grad_sampling_location, float *grad_weights) {
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride = num_embeds;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const float top_grad_mc_ms_feat = grad_output * weight;
  float grad_h_weight = 0, grad_w_weight = 0;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_mc_ms_feat + ptr1, w1 * top_grad_mc_ms_feat);
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_mc_ms_feat + ptr2, w2 * top_grad_mc_ms_feat);
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_mc_ms_feat + ptr3, w3 * top_grad_mc_ms_feat);
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_mc_ms_feat + ptr4, w4 * top_grad_mc_ms_feat);
  }

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  atomicAdd(grad_weights, grad_output * val);
  atomicAdd(grad_sampling_location, width * grad_w_weight * top_grad_mc_ms_feat);
  atomicAdd(grad_sampling_location + 1, height * grad_h_weight * top_grad_mc_ms_feat);
}


__global__ void deformable_aggregation_kernel(
    const int num_kernels,
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    const float weight = *(weights + idx / (num_embeds / num_groups));
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int scale_index = idx % num_scale;
    idx /= num_scale;

    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;

    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx % batch_size;
    idx /= batch_size;

    anchor_index = batch_index * num_anchors + anchor_index;
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) return;
    
    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    atomicAdd(
        output + anchor_index * num_embeds + channel_index,
        bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight
    );
}

__global__ void deformable_aggregation_kernel_opt(
    float *output,
    const float *mc_ms_feat,
    const int2 *spatial_shape,
    const int *scale_start_index,
    const float2 *sample_location,
    const float *weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups)
{
  const int channel_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int anchor_index = blockIdx.y * blockDim.y + threadIdx.y;

  if (channel_index >= num_embeds || anchor_index >= num_anchors)
    return;

  const int group_size = num_embeds / num_groups;
  const int group_index = channel_index / group_size;

  float sum = 0;
  for (int batch_index = 0; batch_index < batch_size; batch_index++)
  {
    const int index0 = batch_index * num_anchors + anchor_index; // (batch_index, anchor_index)
    for (int pts_index = 0; pts_index < num_pts; pts_index++)
    {
      const int index1 = index0 * num_pts + pts_index; // (batch_index, anchor_index, pts_index)
      for (int cam_index = 0; cam_index < num_cams; cam_index++)
      {
        const int index2 = index1 * num_cams + cam_index; // (batch_index, anchor_index, pts_index, cam_index)
        const float2 loc = sample_location[index2];
        if (loc.x <= 0 || loc.x >= 1 || loc.y <= 0 || loc.y >= 1)
          continue;

        for (int scale_index = 0; scale_index < num_scale; scale_index++)
        {
          const int index3 = index2 * num_scale + scale_index; // (batch_index, anchor_index, pts_index, cam_index, scale_index)
          const int index4 = index3 * num_groups + group_index;
          const float weight = weights[index4];

          const int cam_scale_index = cam_index * num_scale + scale_index;
          const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;
          const int2 shape = spatial_shape[cam_scale_index];
          const int h = shape.x;
          const int w = shape.y;

          const float h_im = loc.y * h - 0.5f;
          const float w_im = loc.x * w - 0.5f;
          const float value = bilinear_sampling_opt(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight;
          sum += value;
        }
      }
    }
  }
  output[anchor_index * num_embeds + channel_index] = sum;
}

__global__ void deformable_aggregation_grad_kernel(
    const int num_kernels,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    const float* grad_output,
    float* grad_mc_ms_feat,
    float* grad_sampling_location,
    float* grad_weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    const int weights_ptr = idx / (num_embeds / num_groups);
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int scale_index = idx % num_scale;
    idx /= num_scale;

    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;

    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx % batch_size;
    idx /= batch_size;

    anchor_index = batch_index * num_anchors + anchor_index;
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) return;
    
    const float grad = grad_output[anchor_index*num_embeds + channel_index];

    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    /* atomicAdd( */
    /*     output + anchor_index * num_embeds + channel_index, */
    /*     bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight */
    /* ); */
    const float weight = weights[weights_ptr];
    float *grad_weights_ptr = grad_weights + weights_ptr;
    float *grad_location_ptr = grad_sampling_location + loc_offset;
    bilinear_sampling_grad(
        mc_ms_feat, weight, h, w, num_embeds, h_im, w_im,
        value_offset,
        grad,
        grad_mc_ms_feat, grad_location_ptr, grad_weights_ptr
    );
}

__device__ __inline__ float3 transform_point(float3 p, float sx, float sy, float sz, float cost, float sint, float tx, float ty, float tz)
{
  p.x *= sx;
  p.y *= sy;
  p.z *= sz;
  const float X = cost * p.x - sint * p.y + tx;
  const float Y = sint * p.x + cost * p.y + ty;
  const float Z = p.z + tz;
  return make_float3(X, Y, Z);
}

__global__ void keypoints_generation_kernel(
  float3* keypoints,
  const float* anchors,
  const float3* fixed_scale,
  const float3* learn_scale,
  int batch_size,
  int num_anchors,
  int num_fixed_pts,
  int num_learn_pts
)
{
  constexpr int ANCHOR_STEP = 11;

  const int anchor_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_index = blockIdx.y;
  if (anchor_index >= num_anchors || batch_index >= batch_size)
    return;

  const int num_pts = num_fixed_pts + num_learn_pts;
  const int anchor_offset = batch_index * num_anchors + anchor_index;

  anchors += anchor_offset * ANCHOR_STEP;
  learn_scale += anchor_offset * num_learn_pts;
  keypoints += anchor_offset * num_pts;

  const float tx = anchors[0];
  const float ty = anchors[1];
  const float tz = anchors[2];
  const float sx = expf(anchors[3]);
  const float sy = expf(anchors[4]);
  const float sz = expf(anchors[5]);
  const float sint = anchors[6];
  const float cost = anchors[7];

  for (int i = 0; i < num_fixed_pts; i++)
    keypoints[i] = transform_point(fixed_scale[i], sx, sy, sz, cost, sint, tx, ty, tz);

  for (int i = 0; i < num_learn_pts; i++)
    keypoints[i + num_fixed_pts] = transform_point(learn_scale[i], sx, sy, sz, cost, sint, tx, ty, tz);
}

__global__ void group_weights_feature_embedding_kernel(
  float* output_feature,
  const float* instance_feature,
  const float* anchor_embed,
  const float* camera_embed,
  int batch_size,
  int num_anchors,
  int num_embeds,
  int num_cams
)
{
  const int channel_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int anchor_index = blockIdx.y * blockDim.y + threadIdx.y;
  const int batch_index = blockIdx.z;
  if (channel_index >= num_embeds || anchor_index >= num_anchors || batch_index >= batch_size)
    return;

  const int anchor_offset = batch_index * num_anchors + anchor_index;
  instance_feature += anchor_offset * num_embeds;
  anchor_embed += anchor_offset * num_embeds;
  output_feature += anchor_offset * num_cams * num_embeds;

  const float tmp_feature = instance_feature[channel_index] + anchor_embed[channel_index];
  for (int cam_index = 0; cam_index < num_cams; cam_index++)
    output_feature[cam_index * num_embeds + channel_index] = camera_embed[cam_index * num_embeds + channel_index] + tmp_feature;
}

__device__ __inline__ float warp_sum(float value)
{
  for (int mask = 16; mask > 0; mask /= 2)
    value += __shfl_xor_sync(0xffffffff, value, mask);
  return value;
}

__device__ __inline__ float warp_max(float value)
{
  for (int mask = 16; mask > 0; mask /=2)
    value = ::max(value, __shfl_xor_sync(0xffffffff, value, mask));
  return value;
}

__device__ __inline__ float block_sum(float value, float* reduce_buffer)
{
  const int warp_index = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  value = warp_sum(value);

  if (lane == 0)
    reduce_buffer[warp_index] = value;
  __syncthreads();

  if (warp_index == 0 && lane == 0)
  {
    for (int i = 1; i < NUM_WARPS_PER_BLOCK; i++)
      reduce_buffer[0] += reduce_buffer[i];
  }
  __syncthreads();

  return reduce_buffer[0];
}

__device__ __inline__ float block_max(float value, float* reduce_buffer)
{
  const int warp_index = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  value = warp_max(value);

  if (lane == 0)
    reduce_buffer[warp_index] = value;
  __syncthreads();

  if (warp_index == 0 && lane == 0)
  {
    for (int i = 1; i < NUM_WARPS_PER_BLOCK; i++)
      reduce_buffer[0] = ::max(reduce_buffer[0], reduce_buffer[i]);
  }
  __syncthreads();

  return reduce_buffer[0];
}

__device__ __inline__ void copy4(float* dst, const float* src, int n)
{
  float4* dst4 = (float4*)dst;
  const float4* src4 = (const float4*)src;
  for (int i = threadIdx.x; i < n / 4; i += blockDim.x)
    dst4[i] = src4[i];
}

__device__ __inline__ void apply_softmax_warp(float* feature, int n)
{
  const int lane = threadIdx.x % WARP_SIZE;

  float maxf = 0;
  for (int i = lane; i < n; i += WARP_SIZE)
    maxf = ::max(maxf, feature[i]);
  maxf = warp_max(maxf);

  float sumf = 0;
  for (int i = lane; i < n; i += WARP_SIZE)
  {
    const float f = expf(feature[i] - maxf);
    feature[i] = f;
    sumf += f;
  }
  sumf = warp_sum(sumf);

  const float scale = 1.f / sumf;
  for (int i = lane; i < n; i += WARP_SIZE)
    feature[i] *= scale;
}

__device__ __inline__ void apply_softmax_unroll(float* feature, int n, float* reduce_buffer)
{
  const int count = n / 4;
  float4* feature4 = (float4*)feature;

  float maxf = 0;
  for (int i = threadIdx.x; i < count; i += blockDim.x)
  {
    const float4 f4 = feature4[i];
    maxf = ::max(maxf, ::max(::max(f4.x, f4.y), ::max(f4.z, f4.w)));
  }
  maxf = block_max(maxf, reduce_buffer);

  float sumf = 0;
  for (int i = threadIdx.x; i < count; i += blockDim.x)
  {
    float4 f4 = feature4[i];
    f4.x = expf(f4.x - maxf);
    f4.y = expf(f4.y - maxf);
    f4.z = expf(f4.z - maxf);
    f4.w = expf(f4.w - maxf);
    feature4[i] = f4;
    sumf += (f4.x + f4.y + f4.z + f4.w);
  }
  sumf = block_sum(sumf, reduce_buffer);

  const float scale = 1.f / sumf;
  for (int i = threadIdx.x; i < count; i += blockDim.x)
  {
    float4 f4 = feature4[i];
    f4.x *= scale;
    f4.y *= scale;
    f4.z *= scale;
    f4.w *= scale;
    feature4[i] = f4;
  }
}

__global__ void group_weights_softmax_kernel(
  float* feature,
  int batch_size,
  int num_anchors,
  int num_classes,
  int num_groups
)
{
  constexpr int NUM_GROUPS = 8;
  constexpr int NUM_CLASSES = 312;

  const int batch_index = blockIdx.y;
  const int anchor_index = blockIdx.x;
  const int anchor_offset = batch_index * num_anchors + anchor_index;
  const int feature_size = num_classes * num_groups;

  feature += anchor_offset * feature_size;

  __shared__ float buffer[NUM_GROUPS][NUM_CLASSES];
  for (int feature_index = threadIdx.x; feature_index < feature_size; feature_index += blockDim.x)
  {
    const int group_index = feature_index % NUM_GROUPS;
    const int class_index = feature_index / NUM_GROUPS;
    buffer[group_index][class_index] = feature[feature_index];
  }
  __syncthreads();

  const int warp_index = threadIdx.x / WARP_SIZE;
  apply_softmax_warp(buffer[warp_index], NUM_CLASSES);
  __syncthreads();

  for (int feature_index = threadIdx.x; feature_index < feature_size; feature_index += blockDim.x)
  {
    const int group_index = feature_index % NUM_GROUPS;
    const int class_index = feature_index / NUM_GROUPS;
    feature[feature_index] = buffer[group_index][class_index];
  }
}

__device__ inline float2 project_points(const float* T, float3 p)
{
  const float X = T[0] * p.x + T[1] * p.y + T[ 2] * p.z + T[ 3];
  const float Y = T[4] * p.x + T[5] * p.y + T[ 6] * p.z + T[ 7];
  const float Z = T[8] * p.x + T[9] * p.y + T[10] * p.z + T[11];
  const float invZ = 1 / ::max(Z, 1e-5f);
  const float u = invZ * X;
  const float v = invZ * Y;
  return make_float2(u, v);
}

__global__ void keypoints_projection_kernel(
  float2* output,
  const float3* keypoints,
  const float* projection_mat,
  const float* image_wh,
  int batch_size,
  int num_anchors,
  int num_pts,
  int num_cams
)
{
  const int anchor_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_index = blockIdx.y;
  if (anchor_index >= num_anchors || batch_index >= batch_size)
    return;

  const int anchor_offset = batch_index * num_anchors + anchor_index;

  keypoints += anchor_offset * num_pts;
  output += anchor_offset * num_pts * num_cams;

  constexpr int NUM_CMAS = 6;
  constexpr int NEL_PROJ = 12;

  __shared__ float P[NUM_CMAS][NEL_PROJ];
  const int warp_index = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  if (warp_index < NUM_CMAS && lane < NEL_PROJ)
  {
    const int xyz_index = lane / 4;
    const float wh = xyz_index < 2 ? image_wh[warp_index * 2 + xyz_index] : 1;
    P[warp_index][lane] = projection_mat[warp_index * 16 + lane] / wh;
  }
  __syncthreads();

  for (int pts_index = 0; pts_index < num_pts; pts_index++)
  {
    const float3 kpt = keypoints[pts_index];
    for (int cam_index = 0; cam_index < num_cams; cam_index++)
    {
      const float* T = P[cam_index];
      output[pts_index * num_cams + cam_index] = project_points(T, kpt);
    }
  }
}

__global__ void mha_softmax_kernel(
  float* feature,
  int batch_size,
  int anchor_size,
  int feature_size
)
{
  const int batch_index = blockIdx.y;
  const int anchor_index = blockIdx.x;
  const int anchor_offset = batch_index * anchor_size + anchor_index;

  feature += anchor_offset * feature_size;

  extern __shared__ float feature_buffer[];
  copy4(feature_buffer, feature, feature_size);
  __syncthreads();

  __shared__ float reduce_buffer[NUM_WARPS_PER_BLOCK];
  apply_softmax_unroll(feature_buffer, feature_size, reduce_buffer);
  __syncthreads();

  copy4(feature, feature_buffer, feature_size);
}

__device__ __inline__ void layer_normalize(float* feature, const float* weight, const float* bias, int n)
{
  const int lane = threadIdx.x % WARP_SIZE;
  float sx = 0, sxx = 0;
  for (int i = lane; i < n; i += WARP_SIZE)
  {
    const float x = feature[i];
    sx += x;
    sxx += x * x;
  }

  sx = warp_sum(sx);
  sxx = warp_sum(sxx);

  const float invn = 1.f / n;
  sx *= invn;
  sxx *= invn;

  const float var = sxx - sx * sx;
  const float scale = 1.f / sqrtf(var + 1e-5f);
  for (int i = lane; i < n; i += WARP_SIZE)
    feature[i] = scale * (feature[i] - sx) * weight[i] + bias[i];
}

__device__ __inline__ void layer_normalize_stable(float* feature, const float* weight, const float* bias, int n)
{
  const int lane = threadIdx.x % WARP_SIZE;

  float mean = 0;
  for (int i = lane; i < n; i += WARP_SIZE)
    mean += feature[i];
  mean = warp_sum(mean);
  mean /= n;

  float var = 0;
  for (int i = lane; i < n; i += WARP_SIZE)
  {
    const float dev =  feature[i] - mean;
    var += dev * dev;
  }
  var = warp_sum(var);
  var /= n;

  const float stddev = sqrtf(var + 1e-5f);
  for (int i = lane; i < n; i += WARP_SIZE)
    feature[i] = ((feature[i] - mean) / stddev) * weight[i] + bias[i];
}

__global__ void elementwise_add_kernel(
  float* dst,
  const float* src1,
  const float* src2,
  int n
)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  dst[i] = src1[i] + src2[i];
}

__global__ void add_bias_kernel(
  float* feature,
  const float* bias,
  int num_features,
  int feature_dim
)
{
  const int feature_index = blockIdx.y * blockDim.y + threadIdx.y;
  if (feature_index >= num_features)
    return;

  feature += feature_index * feature_dim;

  for (int i = threadIdx.x; i < feature_dim; i += WARP_SIZE)
    feature[i] += bias[i];
}

__global__ void bias_relu_kernel(
  float* feature,
  const float* bias,
  int num_features,
  int feature_dim
)
{
  const int feature_index = blockIdx.y * blockDim.y + threadIdx.y;
  if (feature_index >= num_features)
    return;

  feature += feature_index * feature_dim;

  for (int i = threadIdx.x; i < feature_dim; i += WARP_SIZE)
    feature[i] = ::max(feature[i] + bias[i], 0.f);
}

__global__ void bias_relu_norm_kernel(
  float* feature,
  const float* linear_bias,
  const float* norm_weight,
  const float* norm_bias,
  int num_features,
  int feature_dim
)
{
  const int feature_index = blockIdx.y * blockDim.y + threadIdx.y;
  if (feature_index >= num_features)
    return;

  feature += feature_index * feature_dim;

  extern __shared__ float buffer[];
  float* sh_feature = buffer + threadIdx.y * feature_dim;
  for (int i = threadIdx.x; i < feature_dim; i += WARP_SIZE)
    sh_feature[i] = ::max(feature[i] + linear_bias[i], 0.f);

  layer_normalize_stable(sh_feature, norm_weight, norm_bias, feature_dim);

  for (int i = threadIdx.x; i < feature_dim; i += WARP_SIZE)
    feature[i] = sh_feature[i];
}

__global__ void bias_scale_kernel(
  float* feature,
  const float* bias,
  const float* scale,
  int num_features,
  int feature_dim
)
{
  const int feature_index = blockIdx.y * blockDim.y + threadIdx.y;
  if (feature_index >= num_features)
    return;

  feature += feature_index * feature_dim;

  for (int i = threadIdx.x; i < feature_dim; i += WARP_SIZE)
    feature[i] = scale[i] * (feature[i] + bias[i]);
}

__global__ void refinement_state_update_kernel(
  float* output,
  const float* anchors,
  const float* time_intervals,
  int batch_size,
  int num_anchors
)
{
  constexpr int REFINE_STATE_NUM = 8;
  constexpr int ANCHOR_STEP = 11;

  const int anchor_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_index = blockIdx.y;
  if (anchor_index >= num_anchors || batch_index >= batch_size)
    return;

  const int anchor_offset = batch_index * num_anchors + anchor_index;
  output += anchor_offset * ANCHOR_STEP;
  anchors += anchor_offset * ANCHOR_STEP;

  const float time_interval = time_intervals[0];
  const float inv_time_interval = 1.f / time_interval;

  for (int i = 0; i < REFINE_STATE_NUM; i++)
    output[i] += anchors[i];
  for (int i = REFINE_STATE_NUM; i < ANCHOR_STEP; i++)
    output[i] = inv_time_interval * output[i] + anchors[i];
}

__global__ void concatenate_features_kernel(
  float* output,
  const float* input1,
  const float* input2,
  int num_features,
  int input_dims
)
{
  const int feature_index = blockIdx.y * blockDim.y + threadIdx.y;
  if (feature_index >= num_features)
    return;

  const int output_dims = 2 * input_dims;
  input1 += feature_index * input_dims;
  input2 += feature_index * input_dims;
  output += feature_index * output_dims;

  float* output1 = output;
  float* output2 = output + input_dims;
  for (int i = threadIdx.x; i < input_dims; i += WARP_SIZE)
  {
    output1[i] = input1[i];
    output2[i] = input2[i];
  }
}

void deformable_aggregation(
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
#ifdef USE_OPTIMIZED_DFA
  const dim3 block(32, 16);
  const dim3 grid(div_up(num_embeds, block.x), div_up(num_anchors, block.y));
  deformable_aggregation_kernel_opt<<<grid, block>>>(output, mc_ms_feat, (const int2 *)spatial_shape, scale_start_index,
    (const float2 *)sample_location, weights, batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups);
#else
  const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
  deformable_aggregation_kernel<<<(int)ceil(((double)num_kernels / 128)), 128>>>(
      num_kernels, output,
      mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
      batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups);
#endif
}


void deformable_aggregation_grad(
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  const float* grad_output,
  float* grad_mc_ms_feat,
  float* grad_sampling_location,
  float* grad_weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_anchors,
  int num_pts,
  int num_groups
) {
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    deformable_aggregation_grad_kernel
        <<<(int)ceil(((double)num_kernels/128)), 128>>>(
        num_kernels,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
    );
}

void keypoints_generation_cuda(
  float* output,
  const float* anchors,
  const float* fixed_scale,
  const float* learn_scale,
  int batch_size,
  int num_anchors,
  int num_fixed_pts,
  int num_learn_pts
)
{
  const dim3 block(512, 1);
  const dim3 grid(div_up(num_anchors, block.x), batch_size);
  keypoints_generation_kernel<<<grid, block>>>((float3*)output, anchors, (const float3*)fixed_scale, (const float3*)learn_scale,
    batch_size, num_anchors, num_fixed_pts, num_learn_pts);
}

void group_weights_feature_embedding(
  float* output,
  const float* instance_feature,
  const float* anchor_embed,
  const float* camera_embed,
  int batch_size,
  int num_anchors,
  int num_embeds,
  int num_cams
)
{
  const dim3 block(256, 2, 1);
  const dim3 grid(div_up(num_embeds, block.x), div_up(num_anchors, block.y), batch_size);
  group_weights_feature_embedding_kernel<<<grid, block>>>(output, instance_feature, anchor_embed, camera_embed,
    batch_size, num_anchors, num_embeds, num_cams);
}

void group_weights_softmax(
  float* feature,
  int batch_size,
  int num_anchors,
  int num_classes,
  int num_groups
)
{
  const dim3 block(WARP_SIZE * num_groups, 1);
  const dim3 grid(num_anchors, batch_size);
  group_weights_softmax_kernel<<<grid, block>>>(feature, batch_size, num_anchors, num_classes, num_groups); 
}

void keypoints_projection_cuda(
  float* output,
  const float* keypoints,
  const float* projection_mat,
  const float* image_wh,
  int batch_size,
  int num_anchors,
  int num_pts,
  int num_cams
)
{
  const dim3 block(512, 1);
  const dim3 grid(div_up(num_anchors, block.x), batch_size);
  keypoints_projection_kernel<<<grid, block>>>((float2*)output, (const float3*)keypoints, projection_mat, image_wh,
    batch_size, num_anchors, num_pts, num_cams);
}

void elementwise_add(
  float* dst,
  const float* src1,
  const float* src2,
  int n
)
{
  const int block = 512;
  const int grid = div_up(n, block);
  elementwise_add_kernel<<<grid, block>>>(dst, src1, src2, n);
}

void add_bias(
  float* output,
  const float* bias,
  int num_features,
  int feature_dim
)
{
  const dim3 block(WARP_SIZE, 8);
  const dim3 grid(1, div_up(num_features, block.y));
  add_bias_kernel<<<grid, block>>>(output, bias, num_features, feature_dim);
}

void bias_relu(
  float* output,
  const float* bias,
  int num_features,
  int feature_dim
)
{
  const dim3 block(WARP_SIZE, 4);
  const dim3 grid(1, div_up(num_features, block.y));
  const size_t shared_size = sizeof(float) * block.y * feature_dim;
  bias_relu_kernel<<<grid, block, shared_size>>>(output, bias, num_features, feature_dim);
}

void bias_relu_norm(
  float* output,
  const float* linear_bias,
  const float* norm_weight,
  const float* norm_bias,
  int num_features,
  int feature_dim
)
{
  const dim3 block(WARP_SIZE, 4);
  const dim3 grid(1, div_up(num_features, block.y));
  const size_t shared_size = sizeof(float) * block.y * feature_dim;
  bias_relu_norm_kernel<<<grid, block, shared_size>>>(output, linear_bias, norm_weight, norm_bias, num_features, feature_dim);
}

void bias_scale(
  float* output,
  const float* bias,
  const float* scale,
  int num_features,
  int feature_dim
)
{
  const dim3 block(WARP_SIZE, 8);
  const dim3 grid(1, div_up(num_features, block.y));
  bias_scale_kernel<<<grid, block>>>(output, bias, scale, num_features, feature_dim);
}

void refinement_state_update(
  float* output,
  const float* anchors,
  const float* time_intervals,
  int batch_size,
  int num_anchors
)
{
  const dim3 block(512, 1);
  const dim3 grid(div_up(num_anchors, block.x), batch_size);
  refinement_state_update_kernel<<<grid, block>>>(output, anchors, time_intervals, batch_size, num_anchors);
}

void concatenate_features(
  float* output,
  const float* input1,
  const float* input2,
  int num_features,
  int input_dims
)
{
  const dim3 block(WARP_SIZE, 8);
  const dim3 grid(1, div_up(num_features, block.y));
  concatenate_features_kernel<<<grid, block>>>(output, input1, input2, num_features, input_dims);
}

void mha_softmax(
  float* feature,
  int batch_size,
  int num_anchors,
  int feature_size
)
{
  const dim3 block(NUM_WARPS_PER_BLOCK * WARP_SIZE, 1);
  const dim3 grid(num_anchors, batch_size);

  mha_softmax_kernel<<<grid, block, feature_size*sizeof(float)>>>(feature, batch_size, num_anchors, feature_size);
}
