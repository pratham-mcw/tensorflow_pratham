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

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/dequantize.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_round_nearest_afz {

const int kTensorNotAllocated = -1;
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

struct OpData {
  int input_dequantized_id = kTensorNotAllocated;
  int output_dequantized_id = kTensorNotAllocated;

  int input_dequantized_index;
  int output_dequantized_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus AllocateTemporaryTensorsIfRequired(TfLiteContext* context,
                                                TfLiteNode* node,
                                                bool is_quantized) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  int temporaries_count = 0;
  if (is_quantized) {
    data->input_dequantized_index = temporaries_count;
    if (data->input_dequantized_id == kTensorNotAllocated) {
      TF_LITE_ENSURE_OK(context, context->AddTensors(
                                     context, 1, &data->input_dequantized_id));
    }
    ++temporaries_count;
    data->output_dequantized_index = temporaries_count;
    if (data->output_dequantized_id == kTensorNotAllocated) {
      TF_LITE_ENSURE_OK(context, context->AddTensors(
                                     context, 1, &data->output_dequantized_id));
    }
    ++temporaries_count;
  }
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(temporaries_count);
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus EvalImpl(const TfLiteTensor* input, TfLiteTensor* output) {
  const int num_elements = NumElements(output);
  const T* input_data = GetTensorData<T>(input);
  T* output_data = GetTensorData<T>(output);
  for (int i = 0; i < num_elements; ++i) {
    output_data[i] = T(std::round(input_data[i]));
  }
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus EvalRoundNearestAFZQuantized(TfLiteContext* context,
                                          TfLiteNode* node, OpData* data,
                                          const TfLiteTensor* input,
                                          TfLiteTensor* output) {
  const double scale = input->params.scale;
  const int32_t zero_point = input->params.zero_point;
  const int num_elements = NumElements(output);
  const T* input_buffer = GetTensorData<T>(input);
  T* output_buffer = GetTensorData<T>(output);
  for (int i = 0; i < num_elements; ++i) {
    float dequantized_value = (input_buffer[i] - zero_point) * scale;
    float rounded_value = std::round(dequantized_value);
    int32_t quantized_value = static_cast<int32_t>(std::round(rounded_value / scale)) + zero_point;
    quantized_value = std::min(std::max(quantized_value, static_cast<int32_t>(std::numeric_limits<T>::min())),
                               static_cast<int32_t>(std::numeric_limits<T>::max()));
    output_buffer[i] = static_cast<T>(quantized_value);
  }
  return TfLiteStatus::kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_EQ(context,HaveSameShapes(input,output),1);
  TF_LITE_ENSURE(context, input->type == kTfLiteFloat32 ||
                              input->type == kTfLiteBFloat16 ||
                              input->type == kTfLiteFloat16 ||
                              input->type == kTfLiteInt16 ||
                              input->type == kTfLiteInt8);

  bool is_quantized = input->quantization.type != kTfLiteNoQuantization;

  if (input->type == kTfLiteInt8 ||
      (input->type == kTfLiteInt16 && is_quantized)) {
    const auto* input_params =
        reinterpret_cast<TfLiteAffineQuantization*>(input->quantization.params);
    const auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
        output->quantization.params);

    TF_LITE_ENSURE(context, input_params != nullptr);
    TF_LITE_ENSURE(context, input_params->scale != nullptr);
    TF_LITE_ENSURE(context, input_params->scale->size > 0);
    TF_LITE_ENSURE(context, input_params->zero_point->size > 0);

    TF_LITE_ENSURE(context, output_params != nullptr);
    TF_LITE_ENSURE(context, output_params->scale != nullptr);
    TF_LITE_ENSURE(context, output_params->scale->size > 0);
    TF_LITE_ENSURE(context, output_params->zero_point->size > 0);

    if (input->type == kTfLiteInt16) {
      TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    }
  }
  TF_LITE_ENSURE_STATUS(
      AllocateTemporaryTensorsIfRequired(context, node, is_quantized));

  if (is_quantized) {
    node->temporaries->data[data->input_dequantized_index] =
        data->input_dequantized_id;
    TfLiteTensor* input_dequantized;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, data->input_dequantized_index,
                                  &input_dequantized));
    input_dequantized->type = kTfLiteFloat32;
    input_dequantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_dequantized->dims, input->dims)) {
      TfLiteIntArray* input_dequantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, input_dequantized,
                                              input_dequantized_size));
    }
    node->temporaries->data[data->output_dequantized_index] =
        data->output_dequantized_id;
    TfLiteTensor* output_dequantized;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, data->output_dequantized_index,
                                  &output_dequantized));
    output_dequantized->type = kTfLiteFloat32;
    output_dequantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(output_dequantized->dims, output->dims)) {
      TfLiteIntArray* output_dequantized_size =
          TfLiteIntArrayCopy(input_dequantized->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, output_dequantized,
                                              output_dequantized_size));
    }
  }
  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TfLiteType data_type = input->type;
  switch (data_type) {
    case kTfLiteFloat32:
      return EvalImpl<float>(input, output);
    case kTfLiteFloat16:
      return EvalImpl<Eigen::half>(input, output);
    case kTfLiteBFloat16:
      return EvalImpl<Eigen::bfloat16>(input, output);
    case kTfLiteInt8:
      return EvalRoundNearestAFZQuantized<int8_t>(context, node, data, input, output);
    case kTfLiteInt16: {
      if (output->quantization.type == kTfLiteNoQuantization)
        return EvalImpl<int16_t>(input, output);
      else
        return EvalRoundNearestAFZQuantized<int16_t>(context, node, data, input, output);
    }
    default: {
      TF_LITE_KERNEL_LOG(
          context, "Type %d is currently not supported by this op.", data_type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace stablehlo_round_nearest_afz

TfLiteRegistration* Register_STABLEHLO_ROUND_NEAREST_AFZ() {
  static TfLiteRegistration r = {
      stablehlo_round_nearest_afz::Init, stablehlo_round_nearest_afz::Free,
      stablehlo_round_nearest_afz::Prepare, stablehlo_round_nearest_afz::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
