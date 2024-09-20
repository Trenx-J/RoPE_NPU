#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

import sys, os
current_dir = "/home/image_data/jinchunxu/AscendC_Custom/RotaryCustom/"
build_path = os.path.join(current_dir, 'build')
sys.path.append(build_path)
import rope_custom

print(">>>>>>>>>>>>[INFO]: Runing test for rope1d ................")
sys.path.append(os.getcwd())
torch.npu.config.allow_internal_format = False

def rotate_half(x):
    dim=int(x.size()[-1]/2)
    x1=x[:,:,:,:dim]
    x2=x[:,:,:,dim:]
    concat=torch.cat([-x2,x1],-1)
    return concat

def verify_op(x, cos ,sin):
    y=rotate_half(x)
    if datatype == torch.bfloat16: 
        x=x.to(torch.float32)
        y=y.to(torch.float32)
        sin=sin.to(torch.float32)   
        cos=cos.to(torch.float32)
        return (x* cos + y* sin).to(torch.bfloat16)
    return x* cos + y* sin

datatype=torch.float32
dtDic={ torch.float16:0, torch.bfloat16:1, torch.float32:2}
shapes=[1,480*20,24,32]
shape=[1,480*20,1,32]
print(">>>>>>>>>>>>[INFO]:Initing inputs, waiting ...............")
x=torch.rand(shapes,device='cpu', dtype=datatype)
sin=torch.rand(shape,device='cpu', dtype=datatype)
cos=torch.rand(shape,device='cpu', dtype=datatype)
x_npu=x.npu()
sin_npu=sin.npu()
cos_npu=cos.npu()

class TestCustomRope(TestCase):
    def test_rope_custom_ops(self): 
        cpuout = verify_op(x, cos, sin)
        output = rope_custom.rope(x_npu, cos_npu, sin_npu,6* 1024,32)
        output=output.cpu()
       
        not_equal_indices = [i for i, (a, b) in enumerate(zip(output[0,:,:,:],cpuout[0,:,:,:])) if not torch.equal(a,b)]
        print(not_equal_indices[:50])
        print(output[0,0,:2,:]==cpuout[0,0,:2,:])   
        print("reference:\n",cpuout[0,0,:2,:])
        print("rope_custom:\n",output[0,0,:2,:])

        # bfloat16验证精准度
        if(dtDic[datatype]==1):
            print(">>>>>>>>>>>>[INFO]: 元素总数: ",shapes[0]*shapes[1]*shapes[2]*shapes[3])
            print(">>>>>>>>>>>>[INFO]: 错误的数量",torch.sum((cpuout==output)!=True))
            print(">>>>>>>>>>>>[INFO]: 错误占比",torch.sum((cpuout==output)!=True)/(shapes[0]*shapes[1]*shapes[2]*shapes[3]))
            epsilon = 1e-10  # 一个很小的数，避免除以零
            y=cpuout
            y_safe = torch.where(y == 0, torch.tensor(epsilon), y)
            error_ratio = torch.abs((y - output) / y_safe)
            threshold_0_1_percent = 0.001  # 0.1%
            threshold_1_percent = 0.01     # 1%
            count_0_1_percent = torch.sum(error_ratio >= threshold_0_1_percent).item()
            count_1_percent = torch.sum( error_ratio >= threshold_1_percent).item()
            count_0_percent = torch.sum(error_ratio > 0).item()
            print(f'>>>>>>>>>>>>[INFO]: 误差比大于0的个数: {count_0_percent}')
            print(f'>>>>>>>>>>>>[INFO]: 误差比大于0.1%的个数: {count_0_1_percent}')
            print(f'>>>>>>>>>>>>[INFO]: 误差比大于1%的个数: {count_1_percent}')

            top_10_errors, top_10_indices = torch.topk(error_ratio.flatten(), 10)
            y_flat = y.flatten()
            pred_flat = output.flatten()
            print('>>>>>>>>>>>>[INFO]: 误差最大的十个数:')
            for i, idx in enumerate(top_10_indices):
                y_value = y_flat[idx].item()
                pred_value = pred_flat[idx].item()
                error_value = top_10_errors[i].item()
                print(f'误差 {i+1}: y = {y_value}, pred = {pred_value}, 误差比 = {error_value}')

        # 其他精度验证精准度
        # self.assertRtolEqual(output, cpuout)

if __name__ == "__main__":
    print(">>>>>>>>>>>>[INFO]: The running shapes       :", shapes)
    print(">>>>>>>>>>>>[INFO]: The datatype             :", datatype)
    print(">>>>>>>>>>>>[INFO]: Running in NPU, waiting ...............")
    run_tests()
    
