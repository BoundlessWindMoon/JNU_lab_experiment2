#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// 评分函数接口
extern __global__ void gemm_naive(float* A, float* B, float* C, int M, int N, int K);

#ifdef __cplusplus
}
#endif