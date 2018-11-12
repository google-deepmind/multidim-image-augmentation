// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "multidim_image_augmentation/platform/types.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

namespace deepmind {
namespace multidim_image_augmentation {
namespace {

template <typename INTYPE, typename OUTTYPE>
class ApplyTabulatedFunctionsOp : public tensorflow::OpKernel {
 public:
  explicit ApplyTabulatedFunctionsOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("offset", &offset_));
    OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor& input = context->input(0);
    int num_channels = input.dim_size(input.dims() - 1);
    auto tabulated_function = context->input(1).tensor<OUTTYPE, 2>();
    OP_REQUIRES(context, tabulated_function.dimension(0) == num_channels,
                tensorflow::errors::InvalidArgument(
                    "incompatible number of channels. The input tensor has ",
                    num_channels, " channels, and there are ",
                    tabulated_function.dimension(0), " tabulated functions"));
    tensorflow::Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    const INTYPE* input_p = input.flat<INTYPE>().data();
    OUTTYPE* output_p = output->flat<OUTTYPE>().data();

    int64 num_pixels = input.NumElements() / num_channels;
    int64 i = 0;
    for (int64 n = 0; n < num_pixels; ++n) {
      for (int channel = 0; channel < num_channels; ++channel) {
        float index_float = (input_p[i] + offset_) * scale_;

        // find the two closest control points
        int index = std::floor(index_float);
        if (index < 0) index = 0;
        if (index > tabulated_function.dimension(1) - 2)
          index = tabulated_function.dimension(1) - 2;

        // compute the linear function between the control points
        OUTTYPE f0 = tabulated_function(channel, index);
        OUTTYPE f1 = tabulated_function(channel, index + 1);
        OUTTYPE m = f1 - f0;

        // apply it to the input value
        float index_rel = index_float - index;
        output_p[i] = static_cast<OUTTYPE>(f0 + m * index_rel);
        ++i;
      }
    }
  }

 private:
  float offset_;
  float scale_;
};

#define REGISTER_KERNEL(INTYPE, OUTTYPE)                               \
  REGISTER_KERNEL_BUILDER(Name("ApplyTabulatedFunctions")              \
                              .Device(tensorflow::DEVICE_CPU)          \
                              .TypeConstraint<INTYPE>("input_type")    \
                              .TypeConstraint<OUTTYPE>("output_type"), \
                          ApplyTabulatedFunctionsOp<INTYPE, OUTTYPE>)
REGISTER_KERNEL(float, float)
REGISTER_KERNEL(float, int64)
REGISTER_KERNEL(float, int32)
REGISTER_KERNEL(float, uint8)
REGISTER_KERNEL(int64, float)
REGISTER_KERNEL(int64, int64)
REGISTER_KERNEL(int64, int32)
REGISTER_KERNEL(int64, uint8)
REGISTER_KERNEL(int32, float)
REGISTER_KERNEL(int32, int64)
REGISTER_KERNEL(int32, int32)
REGISTER_KERNEL(int32, uint8)
REGISTER_KERNEL(uint8, float)
REGISTER_KERNEL(uint8, int64)
REGISTER_KERNEL(uint8, int32)
REGISTER_KERNEL(uint8, uint8)
#undef REGISTER_KERNEL

}  // namespace
}  // namespace multidim_image_augmentation
}  // namespace deepmind
