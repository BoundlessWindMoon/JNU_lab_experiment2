#include "../include/kernel.h"
Kernel get_kernel(int M, int N, int K) {
// ========================== 实验代码区域开始：算子配置 ==========================
    // gemm_naive:
    int M_tile = 16;
    int N_tile = 16;
    int block_x = 16;
    int block_y = 16;

    // gemm_v0:
    // int M_tile = 64;
    // int N_tile = 64;
    // int block_x = 8;
    // int block_y = 8;

    // gemm_v1 && gemm_v2
    // int M_tile = 128;
    // int N_tile = 128;
    // int block_x = 16;
    // int block_y = 16;

    KernelFunc kernel_func = gemm_naive;
    dim3 block(block_x, block_y);
    dim3 grid((N + N_tile - 1) / N_tile, (M + M_tile - 1) / M_tile);
// ========================== 实验代码区域结束：算子配置 ==========================
    
    Kernel kernel = Kernel(kernel_func, block, grid);
    return kernel;
}