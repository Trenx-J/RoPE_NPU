import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

import sys, os
current_dir = "/home/image_data/jinchunxu/AscendC_Custom/Rope_Custom/"
build_path = os.path.join(current_dir, 'build')
sys.path.append(build_path)
sys.path.append(os.getcwd())
import rope_custom

print(">>>>>>>>>>>>[INFO]: Runing test for rope3d ................")
torch.npu.config.allow_internal_format = False


def rotate_half(x):
    dim=int(x.size()[-1]/2)
    x1=x[:,:,:,:dim]
    x2=x[:,:,:,dim:]
    concat=torch.cat([-x2,x1],-1)
    return concat
def rope1d(tokens, cos, sin):
    return tokens* cos+ rotate_half(tokens) * sin
def op(tokens, cos,sin):
    t, y, x = tokens.chunk(3, dim=-1)
    cos=cos.chunk(3, dim=-1)
    sin=sin.chunk(3, dim=-1)
    t = rope1d(t, cos[0], sin[0])
    y = rope1d(y, cos[1], sin[1])
    x = rope1d(x, cos[2], sin[2])
    tokens = torch.cat((t, y, x), dim=-1)
    return tokens

datatype=torch.float16
t,h,w=(24,30,40)
x = torch.rand((2,28800,24,96), dtype=datatype)
cos = torch.rand((2,28800,1,96), dtype=datatype)
sin = torch.rand((2,28800,1,96), dtype=datatype)
x_npu=x.npu()
cos_npu=cos.npu()
sin_npu=sin.npu()

class TestCustomRope(TestCase):
    def test_rope_custom_ops(self): 
        cpuout = op(x, cos,sin)
        output = rope_custom.rope3d(x_npu, cos_npu, sin_npu, 9* 1024)

        output=output.cpu()
        not_equal_indices = [i for i, (a, b) in enumerate(zip(output[0,:,:,:],cpuout[0,:,:,:])) if not torch.equal(a,b)]
        print(not_equal_indices[:50])
        print(output[0,0,:2,:]==cpuout[0,0,:2,:])   
        print("reference:\n",cpuout[0,0,:2,:])
        print("reuslt:\n",output[0,0,:2,:])
        
        #self.assertRtolEqual(output, cpuout)

if __name__ == "__main__":
    run_tests()