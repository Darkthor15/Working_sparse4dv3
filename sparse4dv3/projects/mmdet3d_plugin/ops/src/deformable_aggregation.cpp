#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

static cublasHandle_t s_handle = 0;

static void init_cublas_handle()
{
  if (s_handle == 0)
  {
    cublasCreate_v2(&s_handle);
    cublasSetPointerMode_v2(s_handle, CUBLAS_POINTER_MODE_HOST);
    cublasSetMathMode(s_handle, CUBLAS_TF32_TENSOR_OP_MATH);
  }
}

void elementwise_add(
  float* dst,
  const float* src1,
  const float* src2,
  int n
);

void add_bias(
  float* output,
  const float* bias,
  int num_features,
  int feature_dim
);

void bias_relu(
  float* output,
  const float* bias,
  int num_features,
  int feature_dim
);

void bias_relu_norm(
  float* output,
  const float* linear_bias,
  const float* norm_weight,
  const float* norm_bias,
  int num_features,
  int feature_dim
);

void bias_scale(
  float* output,
  const float* bias,
  const float* scale,
  int num_features,
  int feature_dim
);

static void matmul(const cublasHandle_t& handle, const float* X, const float* W, float* Y,
  int num_features, int src_feature_dim, int dst_feature_dim, int ldX = -1, int ldW = -1, int ldY = -1)
{
  // X: (num_features, src_features_dim)
  // W: (dst_feature_dim, src_feature_dim)
  // Y: (num_features, dst_feature_dim)
  const float alpha = 1.f;
  const float beta = 0.f;
  const cublasOperation_t transa = CUBLAS_OP_T; // WT
  const cublasOperation_t transb = CUBLAS_OP_N;

  if (ldX < 0) ldX = src_feature_dim;
  if (ldW < 0) ldW = src_feature_dim;
  if (ldY < 0) ldY = dst_feature_dim;

  cublasSgemm_v2(handle, transa, transb, dst_feature_dim, num_features, src_feature_dim,
    &alpha,
    W, ldW,
    X, ldX,
    &beta,
    Y, ldY);
}

static void batched_matmul(const cublasHandle_t& handle, const float* X, const float* W, float* Y,
  int num_features, int src_feature_dim, int dst_feature_dim, int batch_size,
  const int64_t* stridesX, const int64_t* stridesW, const int64_t* stridesY, bool transW, float alpha = 1.f)
{
  // X: (batch_size, num_features, src_features_dim)
  // W: (batch_size, src_features_dim, dst_features_dim)
  // Y: (batch_size, num_features, dst_features_dim)
  // const float alpha = 1.f;
  const float beta = 0.f;
  const cublasOperation_t transa = transW ? CUBLAS_OP_T : CUBLAS_OP_N; // W.T
  const cublasOperation_t transb = CUBLAS_OP_N;

  cublasSgemmStridedBatched(handle, transa, transb, dst_feature_dim, num_features, src_feature_dim,
    &alpha,
    W, stridesW[1], stridesW[0],
    X, stridesX[1], stridesX[0],
    &beta,
    Y, stridesY[1], stridesY[0],
    batch_size);
}

static void linear(const cublasHandle_t& handle, const float* X, const float* W, const float* B, float* Y,
  int num_features, int src_feature_dim, int dst_feature_dim)
{
  matmul(handle, X, W, Y, num_features, src_feature_dim, dst_feature_dim);
  add_bias(Y, B, num_features, dst_feature_dim);
}

static void linear_scale(const cublasHandle_t& handle, const float* X, const float* W, const float* B, const float* S, float* Y,
  int num_features, int src_feature_dim, int dst_feature_dim)
{
  matmul(handle, X, W, Y, num_features, src_feature_dim, dst_feature_dim);
  bias_scale(Y, B, S, num_features, dst_feature_dim);
}

static void linear_relu(const cublasHandle_t& handle, const float* X, const float* W, const float* B, float* Y,
  int num_features, int src_feature_dim, int dst_feature_dim)
{
  matmul(handle, X, W, Y, num_features, src_feature_dim, dst_feature_dim);
  bias_relu(Y, B, num_features, dst_feature_dim);
}

static void linear_relu_norm(const cublasHandle_t& handle, const float* X, const float* LW, const float* LB, const float* NW, const float* NB,
  float* Y, int num_features, int src_feature_dim, int dst_feature_dim)
{
  matmul(handle, X, LW, Y, num_features, src_feature_dim, dst_feature_dim);
  bias_relu_norm(Y, LB, NW, NB, num_features, dst_feature_dim);
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
);
  

/* feat: bs, num_feat, c */
/* _spatial_shape: cam, scale, 2 */
/* _scale_start_index: cam, scale */
/* _sampling_location: bs, anchor, pts, cam, 2 */
/* _weights: bs, anchor, pts, cam, scale, group */
/* output: bs, anchor, c */
/* kernel: bs, anchor, pts, c */


at::Tensor deformable_aggregation_forward(
  const at::Tensor &_mc_ms_feat,
  const at::Tensor &_spatial_shape,
  const at::Tensor &_scale_start_index,
  const at::Tensor &_sampling_location,
  const at::Tensor &_weights
) {
  at::DeviceGuard guard(_mc_ms_feat.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  int batch_size = _mc_ms_feat.size(0);
  int num_feat = _mc_ms_feat.size(1);
  int num_embeds = _mc_ms_feat.size(2);
  int num_cams = _spatial_shape.size(0);
  int num_scale = _spatial_shape.size(1);
  int num_anchors = _sampling_location.size(1);
  int num_pts = _sampling_location.size(2);
  int num_groups = _weights.size(5);

  const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
  const int* spatial_shape = _spatial_shape.data_ptr<int>();
  const int* scale_start_index = _scale_start_index.data_ptr<int>();
  const float* sampling_location = _sampling_location.data_ptr<float>();
  const float* weights = _weights.data_ptr<float>();

  auto output = at::zeros({batch_size, num_anchors, num_embeds}, _mc_ms_feat.options());
  deformable_aggregation(
    output.data_ptr<float>(),
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
    batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
  );
  return output;
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
);


void deformable_aggregation_backward(
  const at::Tensor &_mc_ms_feat,
  const at::Tensor &_spatial_shape,
  const at::Tensor &_scale_start_index,
  const at::Tensor &_sampling_location,
  const at::Tensor &_weights,
  const at::Tensor &_grad_output,
  at::Tensor &_grad_mc_ms_feat,
  at::Tensor &_grad_sampling_location,
  at::Tensor &_grad_weights
) {
  at::DeviceGuard guard(_mc_ms_feat.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  int batch_size = _mc_ms_feat.size(0);
  int num_feat = _mc_ms_feat.size(1);
  int num_embeds = _mc_ms_feat.size(2);
  int num_cams = _spatial_shape.size(0);
  int num_scale = _spatial_shape.size(1);
  int num_anchors = _sampling_location.size(1);
  int num_pts = _sampling_location.size(2);
  int num_groups = _weights.size(5);

  const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
  const int* spatial_shape = _spatial_shape.data_ptr<int>();
  const int* scale_start_index = _scale_start_index.data_ptr<int>();
  const float* sampling_location = _sampling_location.data_ptr<float>();
  const float* weights = _weights.data_ptr<float>();
  const float* grad_output = _grad_output.data_ptr<float>();

  float* grad_mc_ms_feat = _grad_mc_ms_feat.data_ptr<float>();
  float* grad_sampling_location = _grad_sampling_location.data_ptr<float>();
  float* grad_weights = _grad_weights.data_ptr<float>();

  deformable_aggregation_grad(
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
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
);

at::Tensor keypoints_generation(
  const at::Tensor &_anchors,
  const at::Tensor &_fixed_scale,
  const at::Tensor &_learn_scale
) {

  at::DeviceGuard guard(_anchors.device());
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(_anchors));
  const int batch_size = _anchors.size(0);
  const int num_anchors = _anchors.size(1);
  const int num_fixed_pts = _fixed_scale.size(0);
  const int num_learn_pts = _learn_scale.size(2) / 3;
  const int num_pts = num_fixed_pts + num_learn_pts;

  const float* anchors = _anchors.data_ptr<float>();
  const float* fixed_scale = _fixed_scale.data_ptr<float>();
  const float* learn_scale = _learn_scale.data_ptr<float>();

  auto output = at::zeros({ batch_size, num_anchors, num_pts, 3 }, _anchors.options());
  keypoints_generation_cuda(output.data_ptr<float>(), anchors, fixed_scale, learn_scale,
    batch_size, num_anchors, num_fixed_pts, num_learn_pts);
  return output;
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
);

void group_weights_softmax(
  float* feature,
  int batch_size,
  int num_anchors,
  int num_classes,
  int num_groups
);

at::Tensor group_weights_generation(
  const at::Tensor &instance_feature,
  const at::Tensor &anchor_embed,
  const at::Tensor &projection_mat,
  const std::vector<at::Tensor> &weights,
  const std::vector<at::Tensor> &biases,
  int output_dims,
  int num_groups
)
{
  at::DeviceGuard guard(instance_feature.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(instance_feature));

  init_cublas_handle();

#define PTR(T) T.data_ptr<float>()

  const int batch_size = instance_feature.size(0);
  const int num_anchors = instance_feature.size(1);
  const int num_embeds = instance_feature.size(2);
  const auto& options = instance_feature.options();
  const int num_cams = projection_mat.size(1);
  const int proj_dims = 12;
  const int proj_stride = projection_mat.stride(1);

  const int num_classes = num_cams * output_dims / num_groups;
  const int num_features = batch_size * num_anchors * num_cams;

  auto tmp_embed = at::empty({ batch_size, num_cams, num_embeds }, options);
  auto camera_embed = at::empty({ batch_size, num_cams, num_embeds }, options);
  auto feature = at::empty({ batch_size, num_anchors, num_cams, num_embeds }, options);
  auto output = at::empty({ batch_size, num_anchors, num_cams, output_dims }, options);

  // camera_encoder
  matmul(s_handle, PTR(projection_mat), PTR(weights[0]), PTR(tmp_embed), num_cams, proj_dims, num_embeds, proj_stride);
  bias_relu_norm(PTR(tmp_embed), PTR(biases[0]), PTR(weights[1]), PTR(biases[1]), num_cams, num_embeds);
  linear_relu_norm(s_handle, PTR(tmp_embed), PTR(weights[2]), PTR(biases[2]), PTR(weights[3]), PTR(biases[3]), PTR(camera_embed),
    num_cams, num_embeds, num_embeds);

  group_weights_feature_embedding(PTR(feature), PTR(instance_feature), PTR(anchor_embed), PTR(camera_embed),
    batch_size, num_anchors, num_embeds, num_cams);

  linear(s_handle, PTR(feature), PTR(weights[4]), PTR(biases[4]), PTR(output), num_features, num_embeds, output_dims);
  group_weights_softmax(PTR(output), batch_size, num_anchors, num_classes, num_groups);

#undef PTR

  return output;
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
);

at::Tensor keypoints_projection(
  const at::Tensor &_keypoints,
  const at::Tensor &_projection_mat,
  const at::Tensor &_image_wh
) {

  at::DeviceGuard guard(_keypoints.device());
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(_keypoints));

  const int batch_size = _keypoints.size(0);
  const int num_anchors = _keypoints.size(1);
  const int num_pts = _keypoints.size(2);
  const int num_cams = _projection_mat.size(1);

  const float* keypoints = _keypoints.data_ptr<float>();
  const float* projection_mat = _projection_mat.data_ptr<float>();
  const float* image_wh = _image_wh.data_ptr<float>();

  auto output = at::empty({ batch_size, num_anchors, num_pts, num_cams, 2 }, _keypoints.options());
  keypoints_projection_cuda(output.data_ptr<float>(), keypoints, projection_mat, image_wh, batch_size, num_anchors, num_pts, num_cams);
  return output;
}

void refinement_state_update(
  float* output,
  const float* anchors,
  const float* time_intervals,
  int batch_size,
  int num_anchors
);

std::vector<at::Tensor> sparse_box3d_refinement(
  const at::Tensor &instance_feature,
  const at::Tensor &anchor,
  const at::Tensor &anchor_embed,
  const at::Tensor &time_interval,
  const std::vector<at::Tensor> &weights,
  const std::vector<at::Tensor> &biases,
  const at::Tensor &scale,
  int embed_dims,
  int output_dims,
  int num_cls,
  bool return_cls
)
{
  init_cublas_handle();

#define PTR(T) T.data_ptr<float>()

  const int batch_size = instance_feature.size(0);
  const int num_anchors = instance_feature.size(1);
  const auto& options = instance_feature.options();
  const int num_features = batch_size * num_anchors;

  auto feature = at::empty(instance_feature.sizes(), options);
  auto embed1 = at::empty({ batch_size, num_anchors, embed_dims }, options);
  auto embed2 = at::empty({ batch_size, num_anchors, embed_dims }, options);
  auto output = at::empty({ batch_size, num_anchors, output_dims }, options);

  elementwise_add(PTR(feature), PTR(instance_feature), PTR(anchor_embed), num_features * feature.size(2));

  for (int k = 0; k < 2; k++)
  {
    const at::Tensor& src = k == 0 ? feature : embed2;
    at::Tensor& tmp = embed1;
    at::Tensor& dst = embed2;
    const at::Tensor* w = &weights[3 * k];
    const at::Tensor* b = &biases[3 * k];
    linear_relu(s_handle, PTR(src), PTR(w[0]), PTR(b[0]), PTR(tmp), num_features, src.size(2), embed_dims);
    linear_relu_norm(s_handle, PTR(tmp), PTR(w[1]), PTR(b[1]), PTR(w[2]), PTR(b[2]), PTR(dst), num_features, embed_dims, embed_dims);
  }
  linear_scale(s_handle, PTR(embed2), PTR(weights[6]), PTR(biases[6]), PTR(scale), PTR(output), num_features, embed_dims, output_dims);
  refinement_state_update(PTR(output), PTR(anchor), PTR(time_interval), batch_size, num_anchors);

  at::Tensor cls, quality;
  if (return_cls)
  {
    cls = at::empty({ batch_size, num_anchors, num_cls }, options);
    quality = at::empty({ batch_size, num_anchors, 2 }, options);

    for (int k = 0; k < 2; k++)
    {
      const at::Tensor& src = k == 0 ? instance_feature : feature;
      at::Tensor& tmp1 = embed1;
      at::Tensor& tmp2 = embed2;
      at::Tensor& dst = k == 0 ? cls : quality;

      const at::Tensor* w = &weights[5 * k + 7];
      const at::Tensor* b = &biases[5 * k + 7];
      linear_relu_norm(s_handle, PTR(src), PTR(w[0]), PTR(b[0]), PTR(w[1]), PTR(b[1]), PTR(tmp1), num_features, src.size(2), embed_dims);
      linear_relu_norm(s_handle, PTR(tmp1), PTR(w[2]), PTR(b[2]), PTR(w[3]), PTR(b[3]), PTR(tmp2), num_features, embed_dims, embed_dims);
      linear(s_handle, PTR(tmp2), PTR(w[4]), PTR(b[4]), PTR(dst), num_features, embed_dims, dst.size(2));
    }
  }

#undef PTR

  return std::vector<at::Tensor> { output, cls, quality };
}

void concatenate_features(
  float* output,
  const float* input1,
  const float* input2,
  int num_features,
  int input_dims
);

void mha_softmax(
  float* feature,
  int batch_size,
  int num_anchors,
  int feature_size
);

at::Tensor graph_model(
  const at::Tensor &query,
  const at::Tensor &key,
  const at::Tensor &value,
  const at::Tensor &query_pos,
  const at::Tensor &key_pos,
  const std::vector<at::Tensor> &weights,
  const std::vector<at::Tensor> &biases,
  int num_heads,
  int head_dim
)
{
  at::DeviceGuard guard(query.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));

  init_cublas_handle();

#define PTR(T) T.data_ptr<float>()
#define PTR0(T, B) T.data_ptr<float>() + B * T.stride(0)
#define STRIDE(T) T.strides().data()

  const int batch_size     = query.size(0); // 1
  const int num_anchors_q  = query.size(1); // 900
  const int input_dims     = query.size(2); // 256
  const int embed_dims     = input_dims * 2; // 512
  const int num_anchors_k  = key.numel() ? key.size(1) : num_anchors_q; // 600 or 900
  const int num_anchors_v  = value.numel() ? value.size(1) : num_anchors_q; // 600 or 900
  const int num_features_q = batch_size * num_anchors_q;
  const int num_features_k = batch_size * num_anchors_k;
  const int num_features_v = batch_size * num_anchors_v;
  const auto& options      = query.options();

  auto output = at::empty({ batch_size, num_anchors_q, input_dims }, options);

  at::Tensor _q, _k, _v;
  _q = at::empty({ batch_size, num_anchors_q, embed_dims }, options);
  concatenate_features(PTR(_q), PTR(query), PTR(query_pos), num_features_q, input_dims);

  if (key.numel())
  {
    _k = at::empty({ batch_size, num_anchors_k, embed_dims }, options);
    concatenate_features(PTR(_k), PTR(key), PTR(key_pos), num_features_k, input_dims);
  }
  else
  {
    _k = _q;
  }

  if (value.numel())
  {
    _v = at::empty({ batch_size, num_anchors_v, embed_dims }, options);
    matmul(s_handle, PTR(value), PTR(weights[0]), PTR(_v), num_features_v, input_dims, embed_dims);
  }
  else
  {
    _v = _k;
  }

  auto q = at::empty(_q.sizes(), options);
  auto k = at::empty(_k.sizes(), options);
  auto v = at::empty(_v.sizes(), options);
  linear(s_handle, PTR(_q), PTR0(weights[1], 0 * embed_dims), PTR0(biases[0], 0 * embed_dims), PTR(q), num_features_q, embed_dims, embed_dims);
  linear(s_handle, PTR(_k), PTR0(weights[1], 1 * embed_dims), PTR0(biases[0], 1 * embed_dims), PTR(k), num_features_k, embed_dims, embed_dims);
  linear(s_handle, PTR(_v), PTR0(weights[1], 2 * embed_dims), PTR0(biases[0], 2 * embed_dims), PTR(v), num_features_v, embed_dims, embed_dims);

  q = q.view({ num_features_q, num_heads, head_dim }).transpose(0, 1);
  k = k.view({ num_features_k, num_heads, head_dim }).transpose(0, 1);
  v = v.view({ num_features_v, num_heads, head_dim }).transpose(0, 1);

  auto w = at::empty({ num_heads, num_anchors_q, num_anchors_k }, options);
  auto x = at::empty({ num_heads, num_anchors_q, head_dim }, options);
  auto y = at::empty({ num_anchors_q, embed_dims }, options);

  const float scale_q = static_cast<float>(1. / sqrt(q.size(-1)));
  batched_matmul(s_handle, PTR(q), PTR(k), PTR(w), num_anchors_q, head_dim, num_anchors_k, num_heads, STRIDE(q), STRIDE(k), STRIDE(w), true, scale_q);
  mha_softmax(PTR(w), num_heads, num_anchors_q, num_anchors_k);
  batched_matmul(s_handle, PTR(w), PTR(v), PTR(x), num_anchors_q, num_anchors_v, head_dim, num_heads, STRIDE(w), STRIDE(v), STRIDE(x), false);

  x = x.transpose(0, 1).reshape({ num_anchors_q, embed_dims });
  linear(s_handle, PTR(x), PTR(weights[2]), PTR(biases[1]), PTR(y), num_anchors_q, embed_dims, embed_dims);
  y = y.view({ num_anchors_q, batch_size, embed_dims });
  y = _q + y.transpose(0, 1);
  matmul(s_handle, PTR(y), PTR(weights[3]), PTR(output), num_features_q, embed_dims, input_dims);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "deformable_aggregation_forward",
    &deformable_aggregation_forward,
    "deformable_aggregation_forward"
  );
  m.def(
    "deformable_aggregation_backward",
    &deformable_aggregation_backward,
    "deformable_aggregation_backward"
  );
  m.def(
    "keypoints_generation",
    &keypoints_generation,
    "keypoints_generation"
  );
  m.def(
    "group_weights_generation",
    &group_weights_generation,
    "group_weights_generation"
  );
  m.def(
    "keypoints_projection",
    &keypoints_projection,
    "keypoints_projection"
  );
  m.def(
    "sparse_box3d_refinement",
    &sparse_box3d_refinement,
    "sparse_box3d_refinement"
  );
  m.def(
    "graph_model",
    &graph_model,
    "graph_model"
  );
}
