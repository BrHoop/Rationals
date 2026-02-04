#include <cstdint>
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/gemmul8.hpp"

extern "C" {

// Opaque descriptor passed from Python.
struct Gemmul8Descriptor {
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  uint32_t num_moduli;
  uint32_t fastmode;
};

static_assert(sizeof(Gemmul8Descriptor) == 56, "Gemmul8Descriptor size mismatch");

static inline cublasHandle_t get_handle(cudaStream_t stream) {
  static thread_local cublasHandle_t handle = nullptr;
  if (!handle) {
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      return nullptr;
    }
  }
  cublasSetStream(handle, stream);
  return handle;
}

// buffers: [A, B, C]
void gemmul8_f32(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  if (opaque_len != sizeof(Gemmul8Descriptor)) {
    return;
  }
  const auto *desc = reinterpret_cast<const Gemmul8Descriptor *>(opaque);
  const float *A = reinterpret_cast<const float *>(buffers[0]);
  const float *B = reinterpret_cast<const float *>(buffers[1]);
  float *C = reinterpret_cast<float *>(buffers[2]);

  cublasHandle_t handle = get_handle(stream);
  if (!handle) return;

  const float alpha = 1.0f;
  const float beta = 0.0f;

  const unsigned num_moduli = desc->num_moduli < 2 ? 2u : desc->num_moduli;
  const bool fastmode = desc->fastmode != 0;

  size_t worksize = gemmul8::workSize<false>(
      static_cast<size_t>(desc->m),
      static_cast<size_t>(desc->n),
      static_cast<size_t>(desc->k),
      num_moduli);

  void *work = nullptr;
  if (worksize > 0) {
    if (cudaMalloc(&work, worksize) != cudaSuccess) {
      return;
    }
  }

  (void)gemmul8::gemm<float>(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      static_cast<size_t>(desc->m),
      static_cast<size_t>(desc->n),
      static_cast<size_t>(desc->k),
      &alpha,
      A,
      static_cast<size_t>(desc->lda),
      B,
      static_cast<size_t>(desc->ldb),
      &beta,
      C,
      static_cast<size_t>(desc->ldc),
      num_moduli,
      fastmode,
      work,
      nullptr,
      nullptr,
      false,
      false,
      false,
      false);

  if (work) cudaFree(work);
}

void gemmul8_f64(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  if (opaque_len != sizeof(Gemmul8Descriptor)) {
    return;
  }
  const auto *desc = reinterpret_cast<const Gemmul8Descriptor *>(opaque);
  const double *A = reinterpret_cast<const double *>(buffers[0]);
  const double *B = reinterpret_cast<const double *>(buffers[1]);
  double *C = reinterpret_cast<double *>(buffers[2]);

  cublasHandle_t handle = get_handle(stream);
  if (!handle) return;

  const double alpha = 1.0;
  const double beta = 0.0;

  const unsigned num_moduli = desc->num_moduli < 2 ? 2u : desc->num_moduli;
  const bool fastmode = desc->fastmode != 0;

  size_t worksize = gemmul8::workSize<false>(
      static_cast<size_t>(desc->m),
      static_cast<size_t>(desc->n),
      static_cast<size_t>(desc->k),
      num_moduli);

  void *work = nullptr;
  if (worksize > 0) {
    if (cudaMalloc(&work, worksize) != cudaSuccess) {
      return;
    }
  }

  (void)gemmul8::gemm<double>(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      static_cast<size_t>(desc->m),
      static_cast<size_t>(desc->n),
      static_cast<size_t>(desc->k),
      &alpha,
      A,
      static_cast<size_t>(desc->lda),
      B,
      static_cast<size_t>(desc->ldb),
      &beta,
      C,
      static_cast<size_t>(desc->ldc),
      num_moduli,
      fastmode,
      work,
      nullptr,
      nullptr,
      false,
      false,
      false,
      false);

  if (work) cudaFree(work);
}

} // extern "C"
