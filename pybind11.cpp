/**
 * @file pybind11.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "aclrtlaunch_rope_custom.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace my_rope {
at::Tensor run_rope_custom(at::Tensor &x, at::Tensor &c, at::Tensor &s, uint32_t &t)
{
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t blockDim = 48;
    uint32_t totalLength = 1;
    uint32_t tilingLength = t;
    uint32_t input_dim = x.sizes()[3];
    for (uint32_t size : x.sizes()) {
        totalLength *= size;
    }    
    if( input_dim == 96)
        tilingLength= (totalLength/ 96)% tilingLength==0? tilingLength: tilingLength/2;
    
    uint8_t datatype;
    if (x.scalar_type() == torch::kFloat16){
        datatype = 0;
    } else if (x.scalar_type() == torch::kFloat32){
        datatype = 2;
    } else if (x.scalar_type() == torch::kBFloat16){
        datatype = 1;
    }

    // // broadcast sin&cos to x
    if(x.sizes()!=c.sizes())
    {
        if(c.sizes()[1]==1)
        {
            c=c.repeat({1,x.sizes()[1],1,1});
            s=s.repeat({1,x.sizes()[1],1,1});
        }
        else
        {
            c=c.repeat({1,1,x.sizes()[2],1});
            s=s.repeat({1,1,x.sizes()[2],1});
        }
    }
    
    at::Tensor result = at::empty_like(x);
    ACLRT_LAUNCH_KERNEL(rope_custom)
    (blockDim, acl_stream, 
    const_cast<void *>(x.storage().data()), 
    const_cast<void *>(c.storage().data()),
    const_cast<void *>(s.storage().data()),
    const_cast<void *>(result.storage().data()), totalLength, input_dim, tilingLength, datatype);
    return result;
    
}
} // namespace my_rope

PYBIND11_MODULE(rope_custom, m)
{
    m.doc() = "rope_custom pybind11 interfaces"; // optional module docstring
    m.def("run_rope_custom", &my_rope::run_rope_custom, "");
}
