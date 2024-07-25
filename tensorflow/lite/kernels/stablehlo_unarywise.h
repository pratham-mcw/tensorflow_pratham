/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_STABLEHLO_UNARYWISE_H_
#define TENSORFLOW_LITE_KERNELS_STABLEHLO_UNARYWISE_H_

#include <cstdint>
#include <vector>

#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

// Indicates the type of the computation performed by the unary-wise op.
enum class ComputationType { kRound };

TfLiteStatus UnarywisePrepare(TfLiteContext* context, TfLiteNode* node);

// A helper function that converts a tensor index into a flat array index.
template <typename IndexType>
static IndexType TensorIndexToFlat(const IndexType* index, const int64_t dims,
                                   const RuntimeShape& shape) {
  // If it's a scalar, just return the index of the first element.
  if (dims == 0) {
    return 0;
  }
  IndexType flat_index = index[0];
  for (int64_t i = 1; i < dims; ++i) {
    flat_index = flat_index * shape.Dims(i) + index[i];
  }
  return flat_index;
}

template <typename DataType, ComputationType computation_type>
inline DataType ApplyComputation(DataType input) {
  if (computation_type == ComputationType::kRound) {
    return std::round(input);
  }
}

// Evaluates this node given the type of the elements in the output_tensor
// and the type of the elements in the input tensor.
template <ComputationType computation_type, typename DataType>
TfLiteStatus EvalWithType(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor, &input_tensor));
  RuntimeShape input_shape = GetTensorShape(input_tensor);
  const DataType* input_data = GetTensorData<DataType>(input_tensor);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  DataType* output_data = GetTensorData<DataType>(output);

  int input_rank = input_tensor->dims->size;
  std::vector<int64_t> index(input_rank, 0);

  do {
    DataType input_value =
        input_data[TensorIndexToFlat(index.data(), input_rank, input_shape)];
    
    output_data[TensorIndexToFlat(index.data(), input_rank, input_shape)] =
        ApplyComputation<DataType, computation_type>(input_value);
  } while (NextIndex(input_rank, input_tensor->dims->data, index.data()));

  return TfLiteStatus::kTfLiteOk;
}

template <ComputationType computation_type>
TfLiteStatus UnarywiseEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor, &input_tensor));

  TfLiteType data_type = input_tensor->type;

  switch (data_type) {
    case kTfLiteFloat16:
      return EvalWithType<computation_type, Eigen::half>(context, node);
    case kTfLiteBFloat16:
      return EvalWithType<computation_type, Eigen::bfloat16>(context, node);
    case kTfLiteFloat32:
      return EvalWithType<computation_type, float>(context, node);
    case kTfLiteFloat64:
      return EvalWithType<computation_type, double>(context, node);
    case kTfLiteInt8:
      return EvalWithType<computation_type, int8_t>(context, node);
    case kTfLiteInt16:
      return EvalWithType<computation_type, int16_t>(context, node);
    case kTfLiteInt32:
      return EvalWithType<computation_type, int32_t>(context, node);
    case kTfLiteInt64:
      return EvalWithType<computation_type, int64_t>(context, node);
    case kTfLiteUInt8:
      return EvalWithType<computation_type, uint8_t>(context, node);
    case kTfLiteUInt16:
      return EvalWithType<computation_type, uint16_t>(context, node);
    case kTfLiteUInt32:
      return EvalWithType<computation_type, uint32_t>(context, node);
    case kTfLiteUInt64:
      return EvalWithType<computation_type, uint64_t>(context, node);
    default:
      TF_LITE_KERNEL_LOG(context, "(Data Type: %s) currently not supported.\n",
                         TfLiteTypeGetName(data_type));
      return TfLiteStatus::kTfLiteError;
  }
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_STABLEHLO_UNARYWISE_H_
