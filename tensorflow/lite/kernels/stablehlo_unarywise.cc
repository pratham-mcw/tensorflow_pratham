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
#include "tensorflow/lite/kernels/stablehlo_unarywise.h"

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {

TfLiteStatus UnarywisePrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor, &input_tensor));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // We need the copy since ResizeTensor takes ownership of output_size.
  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input_tensor->dims);
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_size));

  return TfLiteStatus::kTfLiteOk;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
