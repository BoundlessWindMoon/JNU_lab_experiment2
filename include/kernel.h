#include <cuda_runtime.h>
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define HALF4(pointer) (reinterpret_cast<half4*>(&(pointer))[0])

typedef void (*KernelFunc)(float*, float*, float*, int, int, int);

class Kernel {
    public:
        Kernel(KernelFunc func, dim3 block, dim3 grid) {
            this->kernel_func = func;
            this->block = block;
            this->grid = grid;
        }
        KernelFunc kernel_func;  
        dim3 block;
        dim3 grid;      
};
    

//算子接口
extern __global__ void gemm_naive(float* A, float* B, float* C, int M, int N, int K);
extern __global__ void gemm_v0(float* A, float* B, float* C, int M, int N, int K);
extern __global__ void gemm_v1(float* A, float* B, float* C, int M, int N, int K);
extern __global__ void gemm_v2(float* A, float* B, float* C, int M, int N, int K);
extern __global__ void gemm_v0_ans(float* A, float* B, float* C, int M, int N, int K);
extern __global__ void gemm_v1_ans(float* A, float* B, float* C, int M, int N, int K);
extern __global__ void gemm_v2_ans(float* A, float* B, float* C, int M, int N, int K);

float get_kernel_rank(Kernel kernel, int M, int N, int K);
Kernel get_kernel(int M, int N, int K);

#ifdef __cplusplus
}
#endif 