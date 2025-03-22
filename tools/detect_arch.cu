#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("sm_%d%d", prop.major, prop.minor);
    return 0;
}
