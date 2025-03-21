// V2性能优化：向量加速
#include "kernel.h"
extern "C" __global__ void gemm_v2(float *A, float *B, float *C, int M, int N, int K) {
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

    int load_a_smem_m = tid >> 1;           
    int load_a_smem_k = (tid & 1) << 2;     
    int load_b_smem_k = tid >> 5;           
    int load_b_smem_n = (tid & 31) << 2;    

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        if (load_a_gmem_k + 3 < K && load_a_gmem_m < M) {
            FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(A[load_a_gmem_addr]);
        } else {
            float4 a_val;
            a_val.x = (load_a_gmem_m < M && load_a_gmem_k < K) ? A[load_a_gmem_addr] : 0.0f;
            a_val.y = (load_a_gmem_m < M && load_a_gmem_k + 1 < K) ? A[load_a_gmem_addr + 1] : 0.0f;
            a_val.z = (load_a_gmem_m < M && load_a_gmem_k + 2 < K) ? A[load_a_gmem_addr + 2] : 0.0f;
            a_val.w = (load_a_gmem_m < M && load_a_gmem_k + 3 < K) ? A[load_a_gmem_addr + 3] : 0.0f;
            FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = a_val;
        }

        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

        // ========================== 实验代码区域开始：向量优化 ==========================
        if (load_b_gmem_n + 3 < N && load_b_gmem_k < K) {
            //FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = ;
        } else {
            // float4 b_val;
            // b_val.x = ;
            // b_val.y = ;
            // b_val.z = ;
            // b_val.w = ;
            // FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = ;
        }
        // ========================== 实验代码区域结束：向量优化 ==========================
        __syncthreads(); 

        // 使用 #pragma unroll 进行循环展开
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    r_c[m][n] += s_a[ty * TM + m][k] * s_b[k][tx * TN + n];     //注意坐标的计算
                }
            }
        }

        __syncthreads(); 
    }

    // ========================== 实验代码区域开始：循环展开 ==========================
    for (int i = 0;i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

            if (store_c_gmem_n + 3 < N && store_c_gmem_m < M) {
                FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
            }
            else {
                if (store_c_gmem_m < M) {
                    for (int k = 0; k < 4; k++) {
                        if (store_c_gmem_n + k < N) {
                            C[store_c_gmem_addr + k] = r_c[i][j + k];
                        }
                    }
                }
            }
        }
    }
    // ========================== 实验代码区域结束：循环展开 ==========================
}