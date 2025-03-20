#include <stdio.h>
#include "rank.h"
#include "kernel.h"

int main() {
    float score = get_kernel_rank(gemm_naive);
    printf("Final Score: %.1f\n", score);
}
