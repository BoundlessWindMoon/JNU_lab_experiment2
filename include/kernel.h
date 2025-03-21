#pragma once
#ifdef __cplusplus
extern "C" {
#endif
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
// 将一个指针转换为 float4 类型的指针并访问它指向的第一个元素
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define HALF4(pointer) (reinterpret_cast<half4*>(&(pointer))[0])


//算子接口
extern __global__ void gemm_naive(float* A, float* B, float* C, int M, int N, int K);
extern __global__ void gemm_v0(float* A, float* B, float* C, int M, int N, int K);
extern __global__ void gemm_v1(float* A, float* B, float* C, int M, int N, int K);
extern __global__ void gemm_v2(float* A, float* B, float* C, int M, int N, int K);
extern __global__ void gemm_v0_ans(float* A, float* B, float* C, int M, int N, int K);
extern __global__ void gemm_v1_ans(float* A, float* B, float* C, int M, int N, int K);
extern __global__ void gemm_v2_ans(float* A, float* B, float* C, int M, int N, int K);
#ifdef __cplusplus
}
#endif