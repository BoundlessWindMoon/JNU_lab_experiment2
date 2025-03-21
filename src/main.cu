#include <stdio.h>
#include "rank.h"
#include "kernel.h"

int main() {
    int BM = 16;
    int BN = 16;
    int TM = 1;
    int TN = 1;
    int block_x = BN / TN;
    int block_y = BM / TM;
    int grid_x = BN;
    int grid_y = BM;

    float score = get_kernel_rank(gemm_naive, block_x, block_y, grid_x, grid_y);
    printf("Final Score: %.1f\n", score);
}
