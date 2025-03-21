#include "kernel.h"
extern "C" __global__ void gemm_smem(float *A, float *B, float *C, int M, int N, int K) {
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

    float r_c[TM][TN] = {0.0};              // 用于暂存C矩阵的数据，存放在寄存器中

    int load_a_smem_m = tid >> 1;           //  tid / 2
    int load_a_smem_k = (tid & 1) << 2;     //  (tid % 2) * 4
    int load_b_smem_k = tid >> 5;           //  tid / 32
    int load_b_smem_n = (tid & 31) << 2;    //  (tid % 32) * 4

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        // block内的线程分工把数据加载到shared memory中，即s_a、s_b
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        // FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(A[load_a_gmem_addr]);

        // 只在边界情况下进行检查，避免所有加载都进行边界判断
        if (load_a_gmem_k + 3 < K && load_a_gmem_m < M) {
            FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(A[load_a_gmem_addr]);
        } else {
            // printf("Thread %d: load_a_gmem_k = %d, load_a_gmem_m = %d, load_a_gmem_addr = %d\n", tid, load_a_gmem_k, load_a_gmem_m, load_a_gmem_addr);
            float4 a_val;
            a_val.x = (load_a_gmem_m < M && load_a_gmem_k < K) ? A[load_a_gmem_addr] : 0.0f;
            a_val.y = (load_a_gmem_m < M && load_a_gmem_k + 1 < K) ? A[load_a_gmem_addr + 1] : 0.0f;
            a_val.z = (load_a_gmem_m < M && load_a_gmem_k + 2 < K) ? A[load_a_gmem_addr + 2] : 0.0f;
            a_val.w = (load_a_gmem_m < M && load_a_gmem_k + 3 < K) ? A[load_a_gmem_addr + 3] : 0.0f;
            FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = a_val;
        }

        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        // FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(B[load_b_gmem_addr]);

        

        if (load_b_gmem_n + 3 < N && load_b_gmem_k < K) {
            FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(B[load_b_gmem_addr]);
        } else {
            float4 b_val;
            b_val.x = (load_b_gmem_n < N && load_b_gmem_k < K) ? B[load_b_gmem_addr] : 0.0f;
            b_val.y = (load_b_gmem_n + 1 < N && load_b_gmem_k < K) ? B[load_b_gmem_addr + 1] : 0.0f;
            b_val.z = (load_b_gmem_n + 2 < N && load_b_gmem_k < K) ? B[load_b_gmem_addr + 2] : 0.0f;
            b_val.w = (load_b_gmem_n + 3 < N && load_b_gmem_k < K) ? B[load_b_gmem_addr + 3] : 0.0f;
            FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = b_val;
        }

        __syncthreads(); //等待所有线程的数据全部加载到shared memory中


        // 每个线程计算TM * TN 子矩阵的结果，并存储到r_c数组寄存器里
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

    // 把计算好的TM * TN子矩阵数组r_c存储到对应C矩阵中
    // int store_c_gemm_m = by * BM + ty * TM;
    #pragma unroll
    for (int i = 0;i < TM; i++) {
        // store_c_gemm_m++;
        int store_c_gmem_m = by * BM + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            // FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);

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
}