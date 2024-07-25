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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite{
namespace {

using ::testing::ElementsAre;

class RoundOpModel : public SingleOpModel {
 public:
  RoundOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_STABLEHLO_ROUND, BuiltinOptions_NONE, 0);
    SetBypassDefaultDelegates();
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input_;
  int output_;
};

TEST(StablehloUnarywise, RoundWorks_Float32) {
  RoundOpModel model({TensorType_FLOAT32, {1, 5}},
                                {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input(), {-2.5f, 0.4f, 0.5f, 0.6f, 2.5f});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAre(-3.0f, 0.0f, 1.0f, 1.0f, 3.0f));
}

}  // namespace
}  // namespace tflite