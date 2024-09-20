/**
 * @file rope_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
constexpr int32_t UNIT_SIZE = 128;

template<class T, class K>
class KernelRope1D {
public:
    __aicore__ inline KernelRope1D() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR c, GM_ADDR s, GM_ADDR z,
                                 uint32_t totalLength, uint32_t input_dim, uint32_t tilingLength)
    {
        this->blockLength = totalLength / GetBlockNum();
        this->tilingLength = tilingLength ;
        this->tileNum = this->blockLength / this->tilingLength ;
        this->input_dim = input_dim;
        this->mask_16 = this->input_dim == 32 ? 0x0000ffff0000ffff : 0x00000000ffffffff;
        this->bufferLength = this->input_dim == 96 ? UNIT_SIZE * this->tilingLength / this->input_dim : this->tilingLength;
        this->mask[0] = this->mask_16;
        this->mask[1] = sizeof(K) == 2 ? this->mask_16: 0;
        xGm.SetGlobalBuffer((__gm__ T *)x + this->blockLength * GetBlockIdx(), this->blockLength);
        cosGm.SetGlobalBuffer((__gm__ T *)c + this->blockLength * GetBlockIdx(), this->blockLength);
        sinGm.SetGlobalBuffer((__gm__ T *)s + this->blockLength * GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ T *)z + this->blockLength * GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->bufferLength * sizeof(T));
        pipe.InitBuffer(inQueueCosSin, BUFFER_NUM, this->bufferLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->bufferLength * sizeof(T));
        if( sizeof(T) != sizeof(K)){
            pipe.InitBuffer(outQueueTemp1, BUFFER_NUM, this->bufferLength * sizeof(K));
            pipe.InitBuffer(outQueueTemp2, BUFFER_NUM, this->bufferLength * sizeof(K));
        }
            
        SetDCParams();
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum ;
        for (int32_t i = 0; i < loopCount; i++) {
            int32_t Offset= i* this->tilingLength;
            CopyIn(Offset);
            if constexpr (std::is_same<T, K>::value)
                Compute(Offset);
            else
                Compute_bf(Offset);
            CopyOut(Offset);
        }
        
    }
private:
    __aicore__ inline void SetDCParams()
    {
        //slice1 for rotate_half(x) datacopy part1
        DCParams[0].blockCount = this->tilingLength/ this->input_dim;
        DCParams[0].blockLen = this->input_dim* sizeof(T)/ 64;
        DCParams[0].srcStride = this->input_dim* sizeof(T)/ 64;
        DCParams[0].dstStride = this->input_dim == 96? (UNIT_SIZE- this->input_dim/ 2)* sizeof(T)/ 32: this->input_dim* sizeof(T)/ 64;

        //slice3 for (x,cos,sin) to copyin and z to copyout
        DCParams[1].blockCount =this->tilingLength/ this->input_dim;
        DCParams[1].blockLen=this->input_dim* sizeof(T)/ 32;
        DCParams[1].srcStride = 0;
        DCParams[1].dstStride = (UNIT_SIZE-this->input_dim)* sizeof(T)/ 32;

        DCParams[2].blockCount = this->tilingLength/ this->input_dim;
        DCParams[2].blockLen = this->input_dim* sizeof(T)/ 32;
        DCParams[2].srcStride = (UNIT_SIZE-this->input_dim)* sizeof(T)/ 32;
        DCParams[2].dstStride = 0;
    }
    __aicore__ inline void CopyIn(int32_t offset)
    {
        LocalTensor<T>  xLocal = inQueueX.AllocTensor<T> ();
        LocalTensor<T>  yLocal = inQueueX.AllocTensor<T> ();
        LocalTensor<T>  cosLocal = inQueueCosSin.AllocTensor<T> ();
        LocalTensor<T>  sinLocal = inQueueCosSin.AllocTensor<T> ();

        if( this->input_dim == 96){
            DataCopy(xLocal, xGm[offset], DCParams[1]);
            DataCopy(cosLocal, cosGm[offset], DCParams[1]);
            DataCopy(sinLocal, sinGm[offset], DCParams[1]);
        }
        else{
            DataCopy(xLocal, xGm[offset], this->tilingLength);
            DataCopy(cosLocal, cosGm[offset], this->tilingLength);
            DataCopy(sinLocal, sinGm[offset], this->tilingLength);

        }   
        DataCopy(yLocal, xGm[offset+ this->input_dim/ 2], DCParams[0]);
        DataCopy(yLocal[this->input_dim/ 2], xGm[offset], DCParams[0]);
        
        inQueueX.EnQue(xLocal);
        inQueueX.EnQue(yLocal);
        inQueueCosSin.EnQue(cosLocal);
        inQueueCosSin.EnQue(sinLocal);
    }
    __aicore__ inline void Compute(int32_t offset)
    {
        LocalTensor<T>  xLocal = inQueueX.DeQue<T> ();
        LocalTensor<T>  yLocal = inQueueX.DeQue<T> ();
        LocalTensor<T>  cosLocal = inQueueCosSin.DeQue<T> ();
        LocalTensor<T>  sinLocal = inQueueCosSin.DeQue<T> ();
        LocalTensor<T>  zLocal = outQueueZ.AllocTensor<T> ();
        
        
        if(this->input_dim == 32 || this->input_dim == 64 )
            Muls(yLocal, yLocal, T(-1), mask, this->tilingLength *sizeof(T)/ 256, {1, 1, 8, 8});
        else
            Muls(yLocal, yLocal, T(-1), this->input_dim/ 2, this->bufferLength/ 128, {1, 1, 4* sizeof(T), 4* sizeof(K)});
        
        // Muls(zLocal, xLocal, T(1), this->tilingLength) ;
        xLocal= xLocal* cosLocal;
        yLocal= yLocal* sinLocal;
        zLocal= xLocal+ yLocal;

        outQueueZ.EnQue<T>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueX.FreeTensor(yLocal);
        inQueueCosSin.FreeTensor(cosLocal);
        inQueueCosSin.FreeTensor(sinLocal);
    }
    __aicore__ inline void Compute_bf(int32_t progress)
    {
        LocalTensor<bfloat16_t> xLocal = inQueueX.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> yLocal = inQueueX.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> cosLocal = inQueueCosSin.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> sinLocal = inQueueCosSin.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> zLocal = outQueueZ.AllocTensor<bfloat16_t>();
        LocalTensor<float> resLocal = outQueueTemp2.AllocTensor<float>();
        LocalTensor<float> Local1 = outQueueTemp1.AllocTensor<float>();
        LocalTensor<float> Local2 = outQueueTemp1.AllocTensor<float>();

        Cast(Local1, xLocal, RoundMode::CAST_NONE, this->bufferLength);
        Cast(Local2, cosLocal, RoundMode::CAST_NONE, this->bufferLength);
        resLocal= Local1* Local2;
        Cast(Local1, yLocal, RoundMode::CAST_NONE, this->bufferLength);
        Cast(Local2, sinLocal, RoundMode::CAST_NONE, this->bufferLength);
        
        if(this->input_dim == 32 || this->input_dim == 64 )
            Muls(Local1, Local1, float(-1), mask, this->tilingLength *sizeof(float)/ 256, {1, 1, 8, 8});  
        else
            Muls(Local1, Local1, float(-1), this->input_dim/ 2, this->bufferLength/ 128, {1, 1, 16, 16});
        
        Local1= Local1* Local2;
        resLocal= Local1+ resLocal;
        Cast(zLocal, resLocal, RoundMode::CAST_ROUND, this->bufferLength);
        outQueueZ.EnQue<bfloat16_t>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueX.FreeTensor(yLocal);
        inQueueCosSin.FreeTensor(cosLocal);
        inQueueCosSin.FreeTensor(sinLocal);
        outQueueTemp1.FreeTensor(Local1);
        outQueueTemp1.FreeTensor(Local2);
        outQueueTemp2.FreeTensor(resLocal);
    }
    __aicore__ inline void CopyOut(int32_t offset)
    {
        LocalTensor<T>  zLocal = outQueueZ.DeQue<T> ();
        if(this->input_dim == 96)
            DataCopy(zGm[offset], zLocal, DCParams[2]);
        else
            DataCopy(zGm[offset], zLocal, this->tilingLength);
        outQueueZ.FreeTensor(zLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueX, inQueueCosSin;
    TQue<QuePosition::VECOUT, 2> outQueueTemp1;
    TQue<QuePosition::VECOUT, 1> outQueueTemp2;
    TQue<QuePosition::VECOUT, 1> outQueueZ;
    GlobalTensor<T> xGm;
    GlobalTensor<T> cosGm;
    GlobalTensor<T> sinGm;
    GlobalTensor<T> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t bufferLength;
    uint32_t input_dim;
    uint32_t tilingLength;
    uint64_t mask_16;
    uint64_t mask[2]={0,0};
    DataCopyParams DCParams[3];

};


template<class T, class K>
class KernelRope3D {
public:
    __aicore__ inline KernelRope3D() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR c, GM_ADDR s, GM_ADDR z,
                                 uint32_t totalLength, uint32_t input_dim, uint32_t tilingLength)
    {
        this->blockLength = totalLength / GetBlockNum();
        this->tilingLength = tilingLength ;
        this->tileNum = this->blockLength / this->tilingLength ;
        this->input_dim = input_dim;
        this->mask_16 = 0x0000ffff0000ffff ;
        this->bufferLength = this->tilingLength;
        this->mask[0] = this->mask_16;
        this->mask[1] = sizeof(K) == 2 ? this->mask_16: 0;
        xGm.SetGlobalBuffer((__gm__ T *)x + this->blockLength * GetBlockIdx(), this->blockLength);
        cosGm.SetGlobalBuffer((__gm__ T *)c + this->blockLength * GetBlockIdx(), this->blockLength);
        sinGm.SetGlobalBuffer((__gm__ T *)s + this->blockLength * GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ T *)z + this->blockLength * GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->bufferLength * sizeof(T));
        pipe.InitBuffer(inQueueCosSin, BUFFER_NUM, this->bufferLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->bufferLength * sizeof(T));
        if( sizeof(T) != sizeof(K)){
            pipe.InitBuffer(outQueueTemp1, BUFFER_NUM, this->bufferLength * sizeof(K));
            pipe.InitBuffer(outQueueTemp2, BUFFER_NUM, this->bufferLength * sizeof(K));
        }
            
        SetDCParams();
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum ;
        for (int32_t i = 0; i < loopCount; i++) {
            for( int32_t j = 0; j < 3; j++ )
            {
                int32_t Offset= i* this->tilingLength+ j* this->input_dim;
                CopyIn(Offset);
                if constexpr (std::is_same<T, K>::value)
                    Compute(Offset);
                else
                    Compute_bf(Offset);
                CopyOut(Offset);
            }
            
        }
        
    }
private:
    __aicore__ inline void SetDCParams()
    {
        //slice1 for rotate_half(x) datacopy part1
        uint64_t offset= 64;
        DCParams[0].blockCount = this->tilingLength/ this->input_dim/ 3;
        DCParams[0].blockLen = this->input_dim/ 2* sizeof(T)/ 32;
        DCParams[0].srcStride = (this->input_dim/ 2 + offset)* sizeof(T)/ 32;
        DCParams[0].dstStride = this->input_dim* sizeof(T)/ 64;

        //slice3 for (x,cos,sin) to copyin and z to copyout
        DCParams[1].blockCount = this->tilingLength/ this->input_dim/ 3;
        DCParams[1].blockLen = this->input_dim* sizeof(T)/ 32;
        DCParams[1].srcStride = offset* sizeof(T)/ 32;
        DCParams[1].dstStride = 0;

        DCParams[2].blockCount = this->tilingLength/ this->input_dim/ 3;
        DCParams[2].blockLen = this->input_dim* sizeof(T)/ 32;
        DCParams[2].srcStride = 0;
        DCParams[2].dstStride = offset* sizeof(T)/ 32;
    }
    __aicore__ inline void CopyIn(int32_t offset)
    {
        LocalTensor<T>  xLocal = inQueueX.AllocTensor<T> ();
        LocalTensor<T>  yLocal = inQueueX.AllocTensor<T> ();
        LocalTensor<T>  cosLocal = inQueueCosSin.AllocTensor<T> ();
        LocalTensor<T>  sinLocal = inQueueCosSin.AllocTensor<T> ();

        DataCopy(xLocal, xGm[offset], DCParams[1]);
        DataCopy(cosLocal, cosGm[offset], DCParams[1]);
        DataCopy(sinLocal, sinGm[offset], DCParams[1]);
        DataCopy(yLocal, xGm[offset+ this->input_dim/ 2], DCParams[0]);
        DataCopy(yLocal[this->input_dim/ 2], xGm[offset], DCParams[0]);
        
        inQueueX.EnQue(xLocal);
        inQueueX.EnQue(yLocal);
        inQueueCosSin.EnQue(cosLocal);
        inQueueCosSin.EnQue(sinLocal);
    }
    __aicore__ inline void Compute(int32_t offset)
    {
        LocalTensor<T>  xLocal = inQueueX.DeQue<T> ();
        LocalTensor<T>  yLocal = inQueueX.DeQue<T> ();
        LocalTensor<T>  cosLocal = inQueueCosSin.DeQue<T> ();
        LocalTensor<T>  sinLocal = inQueueCosSin.DeQue<T> ();
        LocalTensor<T>  zLocal = outQueueZ.AllocTensor<T> ();
        
        
        if(this->input_dim == 32 || this->input_dim == 64 )
            Muls(yLocal, yLocal, T(-1), mask, this->tilingLength *sizeof(T)/ 256, {1, 1, 8, 8});
        else
            Muls(yLocal, yLocal, T(-1), this->input_dim/ 2, this->bufferLength/ 128, {1, 1, 4* sizeof(T), 4* sizeof(K)});
        
        // Muls(zLocal, xLocal, T(1), this->tilingLength) ;
        xLocal= xLocal* cosLocal;
        yLocal= yLocal* sinLocal;
        zLocal= xLocal+ yLocal;

        outQueueZ.EnQue<T>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueX.FreeTensor(yLocal);
        inQueueCosSin.FreeTensor(cosLocal);
        inQueueCosSin.FreeTensor(sinLocal);
    }
    __aicore__ inline void Compute_bf(int32_t progress)
    {
        LocalTensor<bfloat16_t> xLocal = inQueueX.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> yLocal = inQueueX.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> cosLocal = inQueueCosSin.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> sinLocal = inQueueCosSin.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> zLocal = outQueueZ.AllocTensor<bfloat16_t>();
        LocalTensor<float> resLocal = outQueueTemp2.AllocTensor<float>();
        LocalTensor<float> Local1 = outQueueTemp1.AllocTensor<float>();
        LocalTensor<float> Local2 = outQueueTemp1.AllocTensor<float>();

        Cast(Local1, xLocal, RoundMode::CAST_NONE, this->bufferLength);
        Cast(Local2, cosLocal, RoundMode::CAST_NONE, this->bufferLength);
        resLocal= Local1* Local2;
        Cast(Local1, yLocal, RoundMode::CAST_NONE, this->bufferLength);
        Cast(Local2, sinLocal, RoundMode::CAST_NONE, this->bufferLength);
        
        if(this->input_dim == 32 || this->input_dim == 64 )
            Muls(Local1, Local1, float(-1), mask, this->tilingLength *sizeof(float)/ 256, {1, 1, 8, 8});  
        else
            Muls(Local1, Local1, float(-1), this->input_dim/ 2, this->bufferLength/ 128, {1, 1, 16, 16});
        
        Local1= Local1* Local2;
        resLocal= Local1+ resLocal;
        Cast(zLocal, resLocal, RoundMode::CAST_ROUND, this->bufferLength);
        outQueueZ.EnQue<bfloat16_t>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueX.FreeTensor(yLocal);
        inQueueCosSin.FreeTensor(cosLocal);
        inQueueCosSin.FreeTensor(sinLocal);
        outQueueTemp1.FreeTensor(Local1);
        outQueueTemp1.FreeTensor(Local2);
        outQueueTemp2.FreeTensor(resLocal);
    }
    __aicore__ inline void CopyOut(int32_t offset)
    {
        LocalTensor<T>  zLocal = outQueueZ.DeQue<T> ();
        DataCopy(zGm[offset], zLocal, DCParams[2]);
        outQueueZ.FreeTensor(zLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueX, inQueueCosSin;
    TQue<QuePosition::VECOUT, 2> outQueueTemp1;
    TQue<QuePosition::VECOUT, 1> outQueueTemp2;
    TQue<QuePosition::VECOUT, 1> outQueueZ;
    GlobalTensor<T> xGm;
    GlobalTensor<T> cosGm;
    GlobalTensor<T> sinGm;
    GlobalTensor<T> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t bufferLength;
    uint32_t input_dim;
    uint32_t tilingLength;
    uint64_t mask_16;
    uint64_t mask[2]={0,0};
    DataCopyParams DCParams[3];

};


extern "C" __global__ __aicore__ void rope_custom(
                    GM_ADDR x, GM_ADDR c, GM_ADDR s, GM_ADDR z, 
                    uint32_t totalLength, uint32_t input_dim,
                    uint32_t tilingLength, uint8_t datatype)
{

    switch (datatype)
    {
        case 0:// float16
        {
            KernelRope1D<half,half> op;
            op.Init(x, c, s, z, totalLength, input_dim, tilingLength);
            op.Process();
            break;  
        }
        case 1:// bfloat16
        {   
            KernelRope1D<bfloat16_t,float> op;
            op.Init(x, c, s, z, totalLength, input_dim, tilingLength);
            op.Process();
            break;
        }
        case 2:// float32
        {
            KernelRope1D<float,float> op;
            op.Init(x, c, s, z, totalLength, input_dim, tilingLength);
            op.Process();
            break;
        }
        case 3:// float16 3d
        {
            KernelRope3D<half,half> op;
            op.Init(x, c, s, z, totalLength, input_dim, tilingLength);
            op.Process();
            break;  
        }
        case 4:// bfloat16 3d
        {   
            KernelRope3D<bfloat16_t,float> op;
            op.Init(x, c, s, z, totalLength, input_dim, tilingLength);
            op.Process();
            break;
        }
        case 5:// float32 3d
        {
            KernelRope3D<float,float> op;
            op.Init(x, c, s, z, totalLength, input_dim, tilingLength);
            op.Process();
            break;
        }
        default:
        {
            KernelRope1D<half,half> op;
            op.Init(x, c, s, z, totalLength, input_dim, tilingLength);
            op.Process();
            break;  
        }

    }
}
