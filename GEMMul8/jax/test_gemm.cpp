#include "gemmul8.hpp"

// 1. Create a handle to the cuBLAS library context
cublasHandle_t cublas_handle;
cublasCreate(&cublas_handle);

// 2. Settings
const unsigned num_moduli = 14u;  // Accuracy knob: 2 <= num_moduli <= 20
const bool fastmode = true;       // true (fast mode) or false (accurate mode)

// 3. Allocate workspace
const size_t worksize = gemmul8::workSize(m, n, k, num_moduli); // calculate required memory (Byte)
void *work;
cudaMalloc(&work, worksize);

// 4. (Optional) Create a vector to store timing breakdown
std::vector<double> time_breakdown(4, 0.0);

// 5. Run emulation
// The function returns a vector with execution times for each phase.
time_breakdown = gemmul8::gemm(cublas_handle,
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               m, n, k,
                               &alpha, devA, lda,
                               devB, ldb,
                               &beta, devC, ldc,
                               num_moduli, fastmode, work);

// 6. Free workspace
cudaFree(work);

// 7. Destroy a handle
cublasDestroy(cublas_andle);