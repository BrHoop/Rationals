
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')

CUDA_PATH ?= /usr/local/cuda
export PATH := $(CUDA_PATH)/bin:$(PATH)
export LD_LIBRARY_PATH := $(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH)

COMPILER := nvcc
LIBS := -lcublas -lcudart -lcuda -lnvidia-ml -ldl ../lib/libgemmul8.a
FLAGS := -std=c++20 -O3 -I../include
ARCH := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

ENV_VAL :=
ifneq ($(GEMMUL8_NUM_MOD_D),)
ENV_VAL += GEMMUL8_NUM_MOD_D=$(GEMMUL8_NUM_MOD_D)
endif
ifneq ($(GEMMUL8_NUM_MOD_S),)
ENV_VAL += GEMMUL8_NUM_MOD_S=$(GEMMUL8_NUM_MOD_S)
endif
ifneq ($(GEMMUL8_FASTMODE_D),)
ENV_VAL += GEMMUL8_FASTMODE_D=$(GEMMUL8_FASTMODE_D)
endif
ifneq ($(GEMMUL8_FASTMODE_S),)
ENV_VAL += GEMMUL8_FASTMODE_S=$(GEMMUL8_FASTMODE_S)
endif
ifneq ($(GEMMUL8_MAX_M),)
ENV_VAL += GEMMUL8_MAX_M=$(GEMMUL8_MAX_M)
endif
ifneq ($(GEMMUL8_MAX_N),)
ENV_VAL += GEMMUL8_MAX_N=$(GEMMUL8_MAX_N)
endif
ifneq ($(GEMMUL8_MAX_K),)
ENV_VAL += GEMMUL8_MAX_K=$(GEMMUL8_MAX_K)
endif
ifneq ($(GEMMUL8_MAX_NUM_MOD),)
ENV_VAL += GEMMUL8_MAX_NUM_MOD=$(GEMMUL8_MAX_NUM_MOD)
endif

CFLAGS=
CFLAGS += -DSGEMM
CFLAGS += -DDGEMM
CFLAGS += -DCGEMM
CFLAGS += -DZGEMM

LD:=../lib/libgemmul8.so
TARGET1 := test
TARGET2 := test_hijack

all: $(TARGET1) $(TARGET2)

$(TARGET1): $(TARGET1).cu
	$(COMPILER) $< $(FLAGS) $(CFLAGS) $(ARCH) -o $@ $(LIBS)

$(TARGET2): $(TARGET2).cu
	$(COMPILER) $< $(FLAGS) $(ARCH) -o $@ $(LIBS)

run-test:
	compute-sanitizer --tool memcheck ./$(TARGET1)

run-test-hijack:
	LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH) LD_PRELOAD=$(LD):$(LD_PRELOAD) compute-sanitizer --tool memcheck ./$(TARGET2)

clean:
	rm -f *.o
	rm -f $(TARGET1) $(TARGET2)
