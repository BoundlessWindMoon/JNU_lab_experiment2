# 编译器和工具
NVCC = nvcc
AR = ar
MKDIR = mkdir -p

# 目录路径
INC_DIR = ./include
KERNEL_DIR = ./kernel
SRC_DIR = ./src
LIB_DIR = ./lib
BUILD_DIR = ./build

# 编译选项
NVCCFLAGS = -O3 -arch=sm_89 -I$(INC_DIR)
ARFLAGS = rcs

# 目标文件
LIB_NAME = rank
LIB_TARGET = $(LIB_DIR)/lib$(LIB_NAME).a
MAIN_TARGET = gemm_evaluator

# 源文件
SRC_MAIN = $(SRC_DIR)/main.cu
KERNELS := $(shell find $(KERNEL_DIR) -name '*.cu')

# 伪目标
.PHONY: all clean

all: $(LIB_TARGET) $(MAIN_TARGET)

# 编译可执行文件
$(MAIN_TARGET): $(SRC_MAIN) $(KERNELS) $(LIB_TARGET)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -L$(LIB_DIR) -l$(LIB_NAME)

clean:
	rm -rf $(LIB_DIR) $(BUILD_DIR) $(MAIN_TARGET)
