#include <stdio.h>
#include "kernel.h"

int main() {
    const int test_sizes[][3] = {
        {512, 512, 512},
        {512, 256, 1024},
        {256, 256, 4096},
        {1024, 1024, 1024}
    };
    const int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    for(int i = 0; i < num_tests; i++) {
        int M = test_sizes[i][0];
        int N = test_sizes[i][1];
        int K = test_sizes[i][2];
        Kernel kernel = get_kernel(M, N, K);
        float score = get_kernel_rank(kernel, M, N, K);
        printf("test case %d get score: %.1f\n", i+1, score);
    }
}
