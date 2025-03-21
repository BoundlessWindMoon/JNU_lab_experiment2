#include <stdio.h>
#include "rank.h"
#include "kernel.h"

int main() {
    // ========================== 实验代码区域开始：分块大小配置 ==========================
        // gemm_naive:
        int BM = 16;
        int BN = 16;
        int TM = 1;
        int TN = 1;

        // gemm_v0:
        // int BM = 64;
        // int BN = 64;
        // int TM = 8;
        // int TN = 8;

        // gemm_v1 && gemm_v2
        // int BM = 128;
        // int BN = 128;
        // int TM = 8;
        // int TN = 8;
    // ========================== 实验代码区域结束：分块大小配置 ==========================
    int block_x = BN / TN;
    int block_y = BM / TM;
    int grid_x = BN;
    int grid_y = BM;

    float score = get_kernel_rank(gemm_naive, block_x, block_y, grid_x, grid_y);
    printf("Final Score: %.1f\n", score);
}
