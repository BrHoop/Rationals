CUDA_HOME ?= /usr/local/cuda
GEMMUL8_DIR ?= ..

NVCC ?= $(CUDA_HOME)/bin/nvcc

INCLUDES := -I$(GEMMUL8_DIR)/include
LIBDIR := $(GEMMUL8_DIR)/lib

CXXFLAGS := -O3 -std=c++17 -Xcompiler -fPIC $(INCLUDES)

TARGET := libgemmul8_jax.so
SRC := gemmul8_jax.cu

LDFLAGS := -shared -L$(LIBDIR) -Wl,-rpath,$(LIBDIR) -lgemmul8 -lcublas -lcudart -lcuda -ldl

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
