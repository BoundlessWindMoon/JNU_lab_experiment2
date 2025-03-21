#pragma once
typedef void (*KernelFunc)(float*, float*, float*, int, int, int);

#ifdef __cplusplus
extern "C" {
#endif

// 评分函数接口
float get_kernel_rank(KernelFunc student_kernel, int BM, int BN, int TM, int TN);

#ifdef __cplusplus
}
#endif

