// V1性能优化：提高算术强度
#include "kernel.h"
extern "C" __global__ void gemm_v1(float *A, float *B, float *C, int M, int N, int K) {
    
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {0.0};              

    // ========================== 实验代码区域开始：提高算术强度 ==========================
    // 此处 load_a_smem_m, load_a_smem_k, load_b_smem_k, load_b_smem_n 用于占位，需要修改！
    int load_a_smem_m = 0;           
    int load_a_smem_k = 0;     
    int load_b_smem_k = 0;           
    int load_b_smem_n = 0;
    // ========================== 实验代码区域结束：提高算术强度 ==========================    

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        // block内的线程分工把数据加载到shared memory中，即s_a、s_b
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        s_a[load_a_smem_m][load_a_smem_k] = (load_a_gmem_m < M && load_a_gmem_k < K) ? A[load_a_gmem_addr] : 0.0f;
        s_a[load_a_smem_m][load_a_smem_k + 1] = (load_a_gmem_m < M && load_a_gmem_k + 1 < K) ? A[load_a_gmem_addr + 1] : 0.0f;
        s_a[load_a_smem_m][load_a_smem_k + 2] = (load_a_gmem_m < M && load_a_gmem_k + 2 < K) ? A[load_a_gmem_addr + 2] : 0.0f;
        s_a[load_a_smem_m][load_a_smem_k + 3] = (load_a_gmem_m < M && load_a_gmem_k + 3 < K) ? A[load_a_gmem_addr + 3] : 0.0f;

        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

        // ========================== 拷贝gemm_v0代码 ==========================
        // s_b[load_b_smem_k][load_b_smem_n] = ;
        // s_b[load_b_smem_k][load_b_smem_n + 1] = ;
        // s_b[load_b_smem_k][load_b_smem_n + 2] = ;
        // s_b[load_b_smem_k][load_b_smem_n + 3] = ;
        // ========================== 拷贝gemm_v0代码 ==========================

        __syncthreads(); 


        // 每个线程计算TM * TN 子矩阵的结果，并存储到r_c数组寄存器里
        for (int k = 0; k < BK; k++) {
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    r_c[m][n] += s_a[ty * TM + m][k] * s_b[k][tx * TN + n];     //注意坐标的计算
                }
            }
        }

        __syncthreads(); 
    }


    for (int i = 0;i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

            C[store_c_gmem_addr + 0] = (store_c_gmem_m < M && store_c_gmem_n + 0 < N) ? r_c[i][j + 0] : 0.0f;
            C[store_c_gmem_addr + 1] = (store_c_gmem_m < M && store_c_gmem_n + 1 < N) ? r_c[i][j + 1] : 0.0f;
            C[store_c_gmem_addr + 2] = (store_c_gmem_m < M && store_c_gmem_n + 2 < N) ? r_c[i][j + 2] : 0.0f;
            C[store_c_gmem_addr + 3] = (store_c_gmem_m < M && store_c_gmem_n + 3 < N) ? r_c[i][j + 3] : 0.0f;

        }
    }
}