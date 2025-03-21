#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../include/rank.h"

// 基准朴素GEMM核函数（正确性验证用）
__global__ void baseline_gemm_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 生成随机矩阵
void generate_random_matrix(float *mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;  // [0, 1)
    }
}

// 验证结果正确性
bool verify_result(float *student_C, float *baseline_C, int size) {
    const float tolerance = 1e-3;
    for (int i = 0; i < size; i++) {
        if (fabs(student_C[i] - baseline_C[i]) > tolerance) {
            printf("wrong result! index %d your result is %f, correct result is %f\n", i, student_C[i], baseline_C[i]);
            return false;
        }
    }
    return true;
}

// 评估函数实现
float get_kernel_rank(KernelFunc student_kernel, int block_x, int block_y, int grid_x, int grid_y) {
    cudaEvent_t start_student, stop_student;
    cudaEvent_t start_baseline, stop_baseline;
    cudaEventCreate(&start_student);
    cudaEventCreate(&stop_student);
    cudaEventCreate(&start_baseline);
    cudaEventCreate(&stop_baseline);

    float total_time_student = 0.0f;
    float total_time_baseline = 0.0f;
    bool is_correct = true;

    // 定义测试矩阵尺寸（示例：不同规模）
    const int test_sizes[][3] = {
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024}
    };
    const int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int t = 0; t < num_tests; t++) {
        int M = test_sizes[t][0], N = test_sizes[t][1], K = test_sizes[t][2];
        float elapsed_time_student, elapsed_time_baseline;

        // 分配内存
        float *h_A = new float[M * K];
        float *h_B = new float[K * N];
        float *h_student_C = new float[M * N];
        float *h_baseline_C = new float[M * N];

        generate_random_matrix(h_A, M * K);
        generate_random_matrix(h_B, K * N);

        // 设备端内存
        float *d_A, *d_B, *d_student_C, *d_baseline_C;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_student_C, M * N * sizeof(float));
        cudaMalloc(&d_baseline_C, M * N * sizeof(float));

        // 拷贝数据到设备
        cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

        // 运行学生核函数并计时
        dim3 block_student(block_x, block_y);
        dim3 grid_student((N + grid_x - 1) / grid_x, (M + grid_y - 1) / grid_y);
        dim3 block_baseline(16, 16);
        dim3 grid_baseline((N + block_baseline.x - 1) / block_baseline.x, (M + block_baseline.y - 1) / block_baseline.y);

        cudaEventRecord(start_student);
        student_kernel<<<grid_student, block_student>>>(d_A, d_B, d_student_C, M, N, K);
        cudaEventRecord(stop_student);
        cudaEventSynchronize(stop_student);
        cudaEventElapsedTime(&elapsed_time_student, start_student, stop_student);
        total_time_student += elapsed_time_student;

        // 运行基准核函数
        cudaEventRecord(start_baseline);
        baseline_gemm_kernel<<<grid_baseline, block_baseline>>>(d_A, d_B, d_baseline_C, M, N, K);
        cudaEventRecord(stop_baseline);
        cudaEventSynchronize(stop_baseline);
        cudaEventElapsedTime(&elapsed_time_baseline, start_baseline, stop_baseline);
        total_time_baseline += elapsed_time_baseline;

        // 拷贝结果回主机
        cudaMemcpy(h_student_C, d_student_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_baseline_C, d_baseline_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        // 检查正确性
        if (!verify_result(h_student_C, h_baseline_C, M * N)) {
            is_correct = false;
            break;
        }

        // 释放内存
        delete[] h_A;
        delete[] h_B;
        delete[] h_student_C;
        delete[] h_baseline_C;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_student_C);
        cudaFree(d_baseline_C);
    }

    cudaEventDestroy(start_student);
    cudaEventDestroy(stop_student);
    cudaEventDestroy(start_baseline);
    cudaEventDestroy(stop_baseline);

    if (!is_correct) return 0.0f;  // 错误直接0分

    // Sigmoid评分逻辑：时间越短分数越高
    float normalized_time = total_time_student / total_time_baseline;
    float score = 40.0f + 60.0f / (1.0f + expf(normalized_time - 1.0f));  // 60分起步，时间越短接近100分
    return fminf(score, 100.0f);  // 确保不超过100分
}




