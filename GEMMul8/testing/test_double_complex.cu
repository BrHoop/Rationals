#include "../include/gemmul8.hpp"
#include "eval.hpp"
#include "getWatt.hpp"
#include "make_matrix.hpp"
#include "self_hipify.hpp"
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#define AVERAGE    100
#define SEED       123456
#define PHI        0.5, 1, 2, 3, 4
#define NUM_MODULI 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
#if defined(GPU_MEM_MB) && GPU_MEM_MB >= 21000
    // more than or equal to 24GB
    #define SIZE 1024, 2048, 4096, 8192, 16384
#else
    #define SIZE 1024, 2048, 4096, 8192
#endif

#if CUBLAS_VER_MAJOR > 13 || (CUBLAS_VER_MAJOR == 13 && CUBLAS_VER_MINOR >= 1) || (CUBLAS_VER_MAJOR == 13 && CUBLAS_VER_PATCH >= 2)
    #define CUBLAS_VERSION_1300u2
#endif

std::string getDeviceName() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::string deviceName = deviceProp.name;

    for (char &c : deviceName) {
        if (c == ' ' || c == '/' || c == '\\') {
            c = '_';
        }
    }
    return deviceName;
}

std::string getCurrentDateTime(std::chrono::system_clock::time_point &now) {
    now                  = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time), "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

#if defined(CUBLAS_VERSION_1300u2)
static inline int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

size_t getFixedPointWorkspaceSizeInBytes(int m, int n, int k, int batchCount, bool isComplex,
                                         cudaEmulationMantissaControl mantissaControl, int maxMantissaBitCount) {

    constexpr double MULTIPLIER = 1.25;

    int mult      = isComplex ? 2 : 1;
    int numSlices = ceildiv(maxMantissaBitCount + 1, 8);

    int padded_m     = ceildiv(m, 1024) * 1024;
    int padded_n     = ceildiv(n, 1024) * 1024;
    int padded_k     = ceildiv(k, 128) * 128;
    int num_blocks_k = ceildiv(k, 64);

    size_t gemm_workspace = sizeof(int8_t) *
                            ((size_t)padded_m * padded_k + (size_t)padded_n * padded_k) * mult * numSlices;

    gemm_workspace += sizeof(int32_t) * ((size_t)padded_m + padded_n) * mult;
    if (isComplex) {
        gemm_workspace += sizeof(double) * (size_t)m * n * mult * mult;
    }

    size_t adp_workspace = 0;
    if (mantissaControl == CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC) {
        adp_workspace = sizeof(int32_t) * ((size_t)m * num_blocks_k + (size_t)n * num_blocks_k + (size_t)m * n) * mult;
    }

    constexpr size_t CONSTANT_SIZE = 128 * 1024 * 1024;
    return (size_t)(std::max(gemm_workspace, adp_workspace) * batchCount * MULTIPLIER) + CONSTANT_SIZE;
}
#endif

void accuracy_check(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_z_accuracy_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    outFile << std::scientific;
    std::cout << std::scientific;
    cublasHandle_t handle;
    cublasCreate(&handle);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    std::vector<double> phi_list{PHI};
    std::vector<size_t> k_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};
    const size_t m = 1024;
    const size_t n = 1024;

    //--------------------
    // workspace
    //--------------------
    const size_t k_max          = *max_element(begin(k_list), end(k_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double2 *work_cpu           = new double2[m * n * 3];
    size_t worksize             = gemmul8::workSize<true>(m, n, k_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, (m * k_max + k_max * n + m * n * 2) * sizeof(double2));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);

#if defined(CUBLAS_VERSION_1300u2)
    cublasHandle_t handle_ozaki1;
    cublasCreate(&handle_ozaki1);
    cudaStream_t stream = NULL;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream(handle_ozaki1, stream);
    cudaEmulationMantissaControl_t mControl = CUDA_EMULATION_MANTISSA_CONTROL_FIXED;
    size_t workspaceSizeInBytes             = getFixedPointWorkspaceSizeInBytes(m, n, k_max, 1, true, mControl, 79);
    void *workspace;
    cudaMalloc(reinterpret_cast<void **>(&workspace), workspaceSizeInBytes);
    cublasSetWorkspace(handle_ozaki1, workspace, workspaceSizeInBytes);
    cublasSetMathMode(handle_ozaki1, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH);
    cublasSetEmulationStrategy(handle_ozaki1, CUBLAS_EMULATION_STRATEGY_EAGER);
    cublasSetFixedPointEmulationMantissaControl(handle_ozaki1, mControl);
#endif

    cudaDeviceSynchronize();

    outFile << "phi,function,";
    std::cout << "phi,function,";
    for (auto &moduli : num_moduli_list) {
        outFile << moduli << ",";
        std::cout << moduli << ",";
    }
    outFile << std::endl;
    std::cout << std::endl;

    for (auto &phi : phi_list) {
        for (auto &k : k_list) {
            double2 *cpuC  = work_cpu;
            double2 *cpuC1 = cpuC + m * n;
            double2 *cpuC2 = cpuC1 + m * n;
            double2 *devA  = reinterpret_cast<double2 *>(work_gpu);
            double2 *devB  = devA + m * k;
            double2 *devC  = devB + k * n;
            double2 *devC1 = devC;
            double2 *devC2 = devC1 + m * n;
            double errmax, errmed;
            std::vector<double> err_OS2_fast;
            std::vector<double> err_OS2_accu;

            //--------------------
            // generate matrices
            //--------------------
            makemat::randmat(m, k, devA, phi, seed);
            makemat::randmat(k, n, devB, phi, seed);

            //--------------------
            // C1+C2 := A*B (double-double arithmetic)
            //--------------------
            eval::dd_gpu::simple_gemm(m, n, k, devA, devB, devC1, devC2);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC1, devC1, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
            cudaMemcpy(cpuC2, devC2, m * n * sizeof(double2), cudaMemcpyDeviceToHost);

            //--------------------
            // C := A*B by FP64
            //--------------------
            double2 alpha{1.0, 0.0};
            double2 beta{0.0, 0.0};
            cudaDeviceSynchronize();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, CUDA_C_64F, m, devB, CUDA_C_64F, k, &beta, devC, CUDA_C_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, errmax, errmed);

            outFile << phi << ",ZGEMMEx (k=" + std::to_string(k) + "),";
            std::cout << phi << ",ZGEMMEx (k=" + std::to_string(k) + "),";
            for (int i = 0; i < num_moduli_list.size(); ++i) {
                outFile << errmax << ",";
                std::cout << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;

            cudaDeviceSynchronize();
            cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                        &alpha, devA, m, devB, k,
                        &beta, devC, m);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, errmax, errmed);

            outFile << phi << ",ZGEMM (k=" + std::to_string(k) + "),";
            std::cout << phi << ",ZGEMM (k=" + std::to_string(k) + "),";
            for (int i = 0; i < num_moduli_list.size(); ++i) {
                outFile << errmax << ",";
                std::cout << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;

#if defined(__NVCC__)
            cudaDeviceSynchronize();
            cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                          &alpha, devA, m, devB, k,
                          &beta, devC, m);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, errmax, errmed);

            outFile << phi << ",ZGEMM3m (k=" + std::to_string(k) + "),";
            std::cout << phi << ",ZGEMM3m (k=" + std::to_string(k) + "),";
            for (int i = 0; i < num_moduli_list.size(); ++i) {
                outFile << errmax << ",";
                std::cout << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;
#endif

#if defined(CUBLAS_VERSION_1300u2)
            for (int mantissaBitCount = 55; mantissaBitCount < 80; mantissaBitCount += 8) {
                cublasSetFixedPointEmulationMaxMantissaBitCount(handle_ozaki1, mantissaBitCount);
                cudaDeviceSynchronize();
                cublasZgemm(handle_ozaki1, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                            &alpha, devA, m, devB, k,
                            &beta, devC, m);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, errmax, errmed);

                outFile << phi << ",OS1-" + std::to_string((mantissaBitCount + 1) / 8) + " (k=" + std::to_string(k) + "),";
                std::cout << phi << ",OS1-" + std::to_string((mantissaBitCount + 1) / 8) + " (k=" + std::to_string(k) + "),";
                for (int i = 0; i < num_moduli_list.size(); ++i) {
                    outFile << errmax << ",";
                    std::cout << errmax << ",";
                }
                outFile << std::endl;
                std::cout << std::endl;
            }
#endif

            //--------------------
            // C := A*B by ozaki-scheme2
            //--------------------
            outFile << phi << ",OS2-fast (k=" + std::to_string(k) + "),";
            std::cout << phi << ",OS2-fast (k=" + std::to_string(k) + "),";
            for (auto &num_moduli : num_moduli_list) {
                std::vector<double> timestmp(4, 0);
                cudaDeviceSynchronize();
                timestmp = gemmul8::gemm<double2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, errmax, errmed);
                outFile << errmax << ",";
                std::cout << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;

            //--------------------
            // C := A*B by ozaki-scheme2
            //--------------------
            outFile << phi << ",OS2-accu (k=" + std::to_string(k) + "),";
            std::cout << phi << ",OS2-accu (k=" + std::to_string(k) + "),";
            for (auto &num_moduli : num_moduli_list) {
                std::vector<double> timestmp(4);
                cudaDeviceSynchronize();
                timestmp = gemmul8::gemm<double2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, errmax, errmed);
                outFile << errmax << ",";
                std::cout << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;
        }
    }

    delete[] work_cpu;
    cudaFree(work_gpu);
    cudaFree(work_gemm);
    cublasDestroy(handle);
    cublasDestroy(handle_ozaki1);
    cudaStreamDestroy(stream);
    cudaDeviceReset();
    outFile.close();
}

void time_check(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_z_time_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    outFile << std::scientific;
    std::cout << std::scientific;
    outFile << "phi,m,n,k,"
            << "function,"
            << "relerr_max,relerr_med,"
            << "TFLOPS,"
            << "total_time [sec],"
            << "conv_64f_2_8i,"
            << "cublasGemmEx,"
            << "conv_32i_2_8u,"
            << "inverse_scaling,"
            << std::endl;
    std::cout << "phi,m,n,k,"
              << "function,"
              << "relerr_max,relerr_med,"
              << "TFLOPS,"
              << "total_time [sec],"
              << "conv_64f_2_8i,"
              << "cublasGemmEx,"
              << "conv_32i_2_8u,"
              << "inverse_scaling,"
              << std::endl;
    cublasHandle_t handle;
    cublasCreate(&handle);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    const double phi        = 0.5;
    std::vector<size_t> n_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};
    const int itermax = AVERAGE;

    //--------------------
    // workspace
    //--------------------
    const size_t n_max          = *max_element(begin(n_list), end(n_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double2 *work_cpu           = new double2[n_max * n_max * 3];
    size_t worksize             = gemmul8::workSize<true>(n_max, n_max, n_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, n_max * n_max * sizeof(double2) * ((num_moduli_max >= 5) ? 3 : 4));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);
    cudaDeviceSynchronize();

    for (auto &n : n_list) {
        size_t m           = n;
        size_t k           = n;
        double2 *cpuC      = work_cpu;
        double2 *cpuC1     = cpuC + m * n;
        double2 *cpuC2     = cpuC1 + m * n;
        double2 *devA      = reinterpret_cast<double2 *>(work_gpu);
        double2 *devB      = devA + m * k;
        double2 *devC      = devB + k * n;
        double2 *devC1     = devC;
        double2 *devC2     = (num_moduli_max >= 5) ? (reinterpret_cast<double2 *>(work_gemm)) : (devC1 + m * n);
        const size_t lda8i = ((k + 15) >> 4) << 4;
        const size_t ldb8i = lda8i;
        int8_t *A8i        = reinterpret_cast<int8_t *>(work_gemm);
        int8_t *B8i        = A8i + lda8i * m;
        int32_t *C32i      = reinterpret_cast<int32_t *>(B8i + ldb8i * n);
        double maxerr = 0.0, mederr = 0.0;
        double time = 0.0;
        std::chrono::system_clock::time_point start, stop;

        //--------------------
        // generate matrices
        //--------------------
        makemat::randmat(m, k, devA, phi, seed);
        makemat::randmat(k, n, devB, phi, seed);

        //--------------------
        // C1+C2 := A*B (double-double arithmetic)
        //--------------------
        eval::dd_gpu::simple_gemm(m, n, k, devA, devB, devC1, devC2);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuC1, devC1, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuC2, devC2, m * n * sizeof(double2), cudaMemcpyDeviceToHost);

        //--------------------
        // C := A*B (int8-TC)
        //--------------------
        makemat::ones(lda8i * m + ldb8i * n, A8i);
        int32_t ialpha = 1;
        int32_t ibeta  = 0;
        cudaDeviceSynchronize();
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, lda8i, &ialpha, A8i, CUDA_R_8I, lda8i, B8i, CUDA_R_8I, ldb8i, &ibeta, C32i, CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        time = 0.0;
        for (int iter = 0; iter < itermax; ++iter) {
            cudaDeviceSynchronize();
            start = std::chrono::system_clock::now();
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, lda8i, &ialpha, A8i, CUDA_R_8I, lda8i, B8i, CUDA_R_8I, ldb8i, &ibeta, C32i, CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            stop = std::chrono::system_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }
        time = time / itermax * 1.e-9;

        outFile << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
        outFile << "," << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                << "," << "," << "," << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
        std::cout << "," << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                  << "," << "," << "," << "," << std::endl;

        //--------------------
        // C := A*B by FP64
        //--------------------
        double2 alpha{1.0, 0.0};
        double2 beta{0.0, 0.0};
        cudaDeviceSynchronize();
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, CUDA_C_64F, m, devB, CUDA_C_64F, k, &beta, devC, CUDA_C_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

        time = 0.0;
        for (int iter = 0; iter < itermax; ++iter) {
            cudaDeviceSynchronize();
            start = std::chrono::system_clock::now();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, CUDA_C_64F, m, devB, CUDA_C_64F, k, &beta, devC, CUDA_C_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            stop = std::chrono::system_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }
        time = time / itermax * 1.e-9;

        outFile << phi << "," << m << "," << n << "," << k << "," << "ZGEMMEx" << ",";
        outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                << "," << "," << "," << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "ZGEMMEx" << ",";
        std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                  << "," << "," << "," << "," << std::endl;

        cudaDeviceSynchronize();
        cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                    &alpha, devA, m, devB, k,
                    &beta, devC, m);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

        time = 0.0;
        for (int iter = 0; iter < itermax; ++iter) {
            cudaDeviceSynchronize();
            start = std::chrono::system_clock::now();
            cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                        &alpha, devA, m, devB, k,
                        &beta, devC, m);
            cudaDeviceSynchronize();
            stop = std::chrono::system_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }
        time = time / itermax * 1.e-9;

        outFile << phi << "," << m << "," << n << "," << k << "," << "ZGEMM" << ",";
        outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                << "," << "," << "," << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "ZGEMM" << ",";
        std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                  << "," << "," << "," << "," << std::endl;

#if defined(__NVCC__)
        cudaDeviceSynchronize();
        cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                      &alpha, devA, m, devB, k,
                      &beta, devC, m);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

        time = 0.0;
        for (int iter = 0; iter < itermax; ++iter) {
            cudaDeviceSynchronize();
            start = std::chrono::system_clock::now();
            cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                          &alpha, devA, m, devB, k,
                          &beta, devC, m);
            cudaDeviceSynchronize();
            stop = std::chrono::system_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }
        time = time / itermax * 1.e-9;

        outFile << phi << "," << m << "," << n << "," << k << "," << "ZGEMM3m" << ",";
        outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                << "," << "," << "," << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "ZGEMM3m" << ",";
        std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                  << "," << "," << "," << "," << std::endl;
#endif

        //--------------------
        // C := A*B by ozaki-scheme2
        //--------------------
        for (auto &num_moduli : num_moduli_list) {
            std::vector<double> times(4, 0);
            std::vector<double> timestmp(4, 0);

            cudaDeviceSynchronize();
            timestmp = gemmul8::gemm<double2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start    = std::chrono::system_clock::now();
                timestmp = gemmul8::gemm<double2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                for (int j = 0; j < 4; ++j) times[j] += timestmp[j];
            }
            time = time / itermax * 1.e-9;
            for (int j = 0; j < 4; ++j) times[j] = times[j] / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
            outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
            std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
        }

        //--------------------
        // C := A*B by ozaki-scheme2
        //--------------------
        for (auto &num_moduli : num_moduli_list) {
            std::vector<double> times(4, 0);
            std::vector<double> timestmp(4);

            cudaDeviceSynchronize();
            timestmp = gemmul8::gemm<double2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start    = std::chrono::system_clock::now();
                timestmp = gemmul8::gemm<double2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                for (int j = 0; j < 4; ++j) times[j] += timestmp[j];
            }
            time = time / itermax * 1.e-9;
            for (int j = 0; j < 4; ++j) times[j] = times[j] / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
        }
    }

    cudaFree(work_gemm);
    cublasDestroy(handle);
    delete[] work_cpu;
    cudaFree(work_gpu);
    outFile.close();
}

void watt_check(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_z_watt_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    outFile << std::scientific;
    std::cout << std::scientific;
    outFile << "phi,m,n,k,"
            << "function,"
            << "relerr_max,relerr_med,"
            << "watt,"
            << "GFLOPS/watt,"
            << std::endl;
    std::cout << "phi,m,n,k,"
              << "function,"
              << "relerr_max,relerr_med,"
              << "watt,"
              << "GFLOPS/watt,"
              << std::endl;
    cublasHandle_t handle;
    cublasCreate(&handle);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    const double phi        = 0.5;
    std::vector<size_t> n_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};

    //--------------------
    // workspace
    //--------------------
    const size_t n_max          = *max_element(begin(n_list), end(n_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double2 *work_cpu           = new double2[n_max * n_max * 3];
    size_t worksize             = gemmul8::workSize<true>(n_max, n_max, n_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, n_max * n_max * 5 * sizeof(double2));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);
    cudaDeviceSynchronize();

    for (auto &n : n_list) {
        size_t m           = n;
        size_t k           = n;
        double2 *cpuC      = work_cpu;
        double2 *cpuC1     = cpuC + m * n;
        double2 *cpuC2     = cpuC1 + m * n;
        double2 *devA      = reinterpret_cast<double2 *>(work_gpu);
        double2 *devB      = devA + m * k;
        double2 *devC      = devB + k * n;
        double2 *devC1     = devC + m * n;
        double2 *devC2     = devC1 + m * n;
        const size_t lda8i = ((k + 15) >> 4) << 4;
        const size_t ldb8i = lda8i;
        int8_t *A8i        = reinterpret_cast<int8_t *>(work_gemm);
        int8_t *B8i        = A8i + lda8i * m;
        int32_t *C32i      = reinterpret_cast<int32_t *>(B8i + ldb8i * n);
        double maxerr = 0.0, mederr = 0.0;

        //--------------------
        // generate matrices
        //--------------------
        makemat::randmat(m, k, devA, phi, seed);
        makemat::randmat(k, n, devB, phi, seed);

        //--------------------
        // C1+C2 := A*B (double-double arithmetic)
        //--------------------
        eval::dd_gpu::simple_gemm(m, n, k, devA, devB, devC1, devC2);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuC1, devC1, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuC2, devC2, m * n * sizeof(double2), cudaMemcpyDeviceToHost);

        //--------------------
        // C := A*B (int8-TC)
        //--------------------
        makemat::ones(lda8i * m + ldb8i * n, A8i);
        int32_t ialpha          = 1;
        int32_t ibeta           = 0;
        std::vector<double> res = getWatt::getWatt(
            [&]() {
                cublasGemmEx(handle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             m,
                             n,
                             lda8i,
                             &ialpha,
                             A8i,
                             CUDA_R_8I,
                             lda8i,
                             B8i,
                             CUDA_R_8I,
                             ldb8i,
                             &ibeta,
                             C32i,
                             CUDA_R_32I,
                             m,
                             CUBLAS_COMPUTE_32I,
                             CUBLAS_GEMM_DEFAULT);
            },
            m,
            n,
            k);

        outFile << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
        outFile << "," << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
        std::cout << "," << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;

        //--------------------
        // C := A*B by FP64
        //--------------------
        double2 alpha{1.0, 0.0};
        double2 beta{0.0, 0.0};
        cudaDeviceSynchronize();
        res = getWatt::getWatt(
            [&]() {
                cublasGemmEx(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             m,
                             n,
                             k,
                             &alpha,
                             devA,
                             CUDA_C_64F,
                             m,
                             devB,
                             CUDA_C_64F,
                             k,
                             &beta,
                             devC,
                             CUDA_C_64F,
                             m,
                             CUBLAS_COMPUTE_64F,
                             CUBLAS_GEMM_DEFAULT);
            },
            m,
            n,
            k);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

        outFile << phi << "," << m << "," << n << "," << k << "," << "ZGEMMEx" << ",";
        outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "ZGEMMEx" << ",";
        std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;

        cudaDeviceSynchronize();
        res = getWatt::getWatt(
            [&]() {
                cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                            &alpha, devA, m, devB, k,
                            &beta, devC, m);
            },
            m,
            n,
            k);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

        outFile << phi << "," << m << "," << n << "," << k << "," << "ZGEMM" << ",";
        outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "ZGEMM" << ",";
        std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;

#if defined(__NVCC__)
        cudaDeviceSynchronize();
        res = getWatt::getWatt(
            [&]() {
                cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                              &alpha, devA, m, devB, k,
                              &beta, devC, m);
            },
            m,
            n,
            k);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

        outFile << phi << "," << m << "," << n << "," << k << "," << "ZGEMM3m" << ",";
        outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "ZGEMM3m" << ",";
        std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
#endif

        //--------------------
        // C := A*B by ozaki-scheme2
        //--------------------
        for (auto &num_moduli : num_moduli_list) {
            cudaDeviceSynchronize();

            res = getWatt::getWatt(
                [&]() {
                    gemmul8::gemm<double2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
                },
                m,
                n,
                k);

            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
            outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
            std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
        }

        //--------------------
        // C := A*B by ozaki-scheme2
        //--------------------
        for (auto &num_moduli : num_moduli_list) {
            cudaDeviceSynchronize();

            res = getWatt::getWatt(
                [&]() {
                    gemmul8::gemm<double2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
                },
                m,
                n,
                k);

            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
        }
    }

    delete[] work_cpu;
    cudaFree(work_gpu);
    cudaFree(work_gemm);
    cublasDestroy(handle);
    outFile.close();
}

void time_check_rect(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_z_time-rect_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    outFile << std::scientific;
    std::cout << std::scientific;
    outFile << "phi,m,n,k,"
            << "function,"
            << "relerr_max,relerr_med,"
            << "TFLOPS,"
            << "total_time [sec],"
            << "conv_64f_2_8i,"
            << "cublasGemmEx,"
            << "conv_32i_2_8u,"
            << "inverse_scaling,"
            << std::endl;
    std::cout << "phi,m,n,k,"
              << "function,"
              << "relerr_max,relerr_med,"
              << "TFLOPS,"
              << "total_time [sec],"
              << "conv_64f_2_8i,"
              << "cublasGemmEx,"
              << "conv_32i_2_8u,"
              << "inverse_scaling,"
              << std::endl;
    cublasHandle_t handle;
    cublasCreate(&handle);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    const double phi        = 0.5;
    std::vector<size_t> n_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};
    const int itermax = AVERAGE;

    //--------------------
    // workspace
    //--------------------
    const size_t n_max          = *max_element(begin(n_list), end(n_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double2 *work_cpu           = new double2[n_max * n_max * 3];
    size_t worksize             = gemmul8::workSize<true>(n_max, n_max, n_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, n_max * n_max * sizeof(double2) * ((num_moduli_max >= 5) ? 3 : 4));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);
    cudaDeviceSynchronize();

    for (auto &n : n_list) {
        size_t m = n;

        for (auto &k : n_list) {
            double2 *cpuC      = work_cpu;
            double2 *cpuC1     = cpuC + m * n;
            double2 *cpuC2     = cpuC1 + m * n;
            double2 *devA      = reinterpret_cast<double2 *>(work_gpu);
            double2 *devB      = devA + m * k;
            double2 *devC      = devB + k * n;
            double2 *devC1     = devC;
            double2 *devC2     = (num_moduli_max >= 5) ? (reinterpret_cast<double2 *>(work_gemm)) : (devC1 + m * n);
            const size_t lda8i = ((k + 15) >> 4) << 4;
            const size_t ldb8i = lda8i;
            int8_t *A8i        = reinterpret_cast<int8_t *>(work_gemm);
            int8_t *B8i        = A8i + lda8i * m;
            int32_t *C32i      = reinterpret_cast<int32_t *>(B8i + ldb8i * n);
            double maxerr = 0.0, mederr = 0.0;
            double time = 0.0;
            std::chrono::system_clock::time_point start, stop;

            //--------------------
            // generate matrices
            //--------------------
            makemat::randmat(m, k, devA, phi, seed);
            makemat::randmat(k, n, devB, phi, seed);

            //--------------------
            // C1+C2 := A*B (double-double arithmetic)
            //--------------------
            eval::dd_gpu::simple_gemm(m, n, k, devA, devB, devC1, devC2);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC1, devC1, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
            cudaMemcpy(cpuC2, devC2, m * n * sizeof(double2), cudaMemcpyDeviceToHost);

            //--------------------
            // C := A*B (int8-TC)
            //--------------------
            makemat::ones(lda8i * m + ldb8i * n, A8i);
            int32_t ialpha = 1;
            int32_t ibeta  = 0;
            cudaDeviceSynchronize();
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, lda8i, &ialpha, A8i, CUDA_R_8I, lda8i, B8i, CUDA_R_8I, ldb8i, &ibeta, C32i, CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start = std::chrono::system_clock::now();
                cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, lda8i, &ialpha, A8i, CUDA_R_8I, lda8i, B8i, CUDA_R_8I, ldb8i, &ibeta, C32i, CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            }
            time = time / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
            outFile << "," << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << "," << "," << "," << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
            std::cout << "," << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << "," << "," << "," << "," << std::endl;

            //--------------------
            // C := A*B by FP64
            //--------------------
            double2 alpha{1.0, 0.0};
            double2 beta{0.0, 0.0};
            cudaDeviceSynchronize();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, CUDA_C_64F, m, devB, CUDA_C_64F, k, &beta, devC, CUDA_C_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start = std::chrono::system_clock::now();
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, CUDA_C_64F, m, devB, CUDA_C_64F, k, &beta, devC, CUDA_C_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            }
            time = time / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "ZGEMMEx" << ",";
            outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << "," << "," << "," << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "ZGEMMEx" << ",";
            std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << "," << "," << "," << "," << std::endl;

            cudaDeviceSynchronize();
            cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                        &alpha, devA, m, devB, k,
                        &beta, devC, m);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start = std::chrono::system_clock::now();
                cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                            &alpha, devA, m, devB, k,
                            &beta, devC, m);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            }
            time = time / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "ZGEMM" << ",";
            outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << "," << "," << "," << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "ZGEMM" << ",";
            std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << "," << "," << "," << "," << std::endl;

#if defined(__NVCC__)
            cudaDeviceSynchronize();
            cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                          &alpha, devA, m, devB, k,
                          &beta, devC, m);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start = std::chrono::system_clock::now();
                cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                              &alpha, devA, m, devB, k,
                              &beta, devC, m);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            }
            time = time / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "ZGEMM3m" << ",";
            outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << "," << "," << "," << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "ZGEMM3m" << ",";
            std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << "," << "," << "," << "," << std::endl;
#endif

            //--------------------
            // C := A*B by ozaki-scheme2
            //--------------------
            for (auto &num_moduli : num_moduli_list) {
                std::vector<double> times(4, 0);
                std::vector<double> timestmp(4, 0);

                cudaDeviceSynchronize();
                timestmp = gemmul8::gemm<double2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

                time = 0.0;
                for (int iter = 0; iter < itermax; ++iter) {
                    cudaDeviceSynchronize();
                    start    = std::chrono::system_clock::now();
                    timestmp = gemmul8::gemm<double2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
                    cudaDeviceSynchronize();
                    stop = std::chrono::system_clock::now();
                    time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                    for (int j = 0; j < 4; ++j) times[j] += timestmp[j];
                }
                time = time / itermax * 1.e-9;
                for (int j = 0; j < 4; ++j) times[j] = times[j] / itermax * 1.e-9;

                outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
                outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                        << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
                std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
                std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                          << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
            }

            //--------------------
            // C := A*B by ozaki-scheme2
            //--------------------
            for (auto &num_moduli : num_moduli_list) {
                std::vector<double> times(4, 0);
                std::vector<double> timestmp(4);

                cudaDeviceSynchronize();
                timestmp = gemmul8::gemm<double2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

                time = 0.0;
                for (int iter = 0; iter < itermax; ++iter) {
                    cudaDeviceSynchronize();
                    start    = std::chrono::system_clock::now();
                    timestmp = gemmul8::gemm<double2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
                    cudaDeviceSynchronize();
                    stop = std::chrono::system_clock::now();
                    time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                    for (int j = 0; j < 4; ++j) times[j] += timestmp[j];
                }
                time = time / itermax * 1.e-9;
                for (int j = 0; j < 4; ++j) times[j] = times[j] / itermax * 1.e-9;

                outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
                outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                        << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
                std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
                std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                          << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
            }
        }
    }

    delete[] work_cpu;
    cudaFree(work_gpu);
    cudaFree(work_gemm);
    cublasDestroy(handle);
    outFile.close();
}

void watt_check_rect(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_z_watt-rect_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    outFile << std::scientific;
    std::cout << std::scientific;
    outFile << "phi,m,n,k,"
            << "function,"
            << "relerr_max,relerr_med,"
            << "watt,"
            << "GFLOPS/watt,"
            << std::endl;
    std::cout << "phi,m,n,k,"
              << "function,"
              << "relerr_max,relerr_med,"
              << "watt,"
              << "GFLOPS/watt,"
              << std::endl;
    cublasHandle_t handle;
    cublasCreate(&handle);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    const double phi        = 0.5;
    std::vector<size_t> n_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};

    //--------------------
    // workspace
    //--------------------
    const size_t n_max          = *max_element(begin(n_list), end(n_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double2 *work_cpu           = new double2[n_max * n_max * 3];
    size_t worksize             = gemmul8::workSize<true>(n_max, n_max, n_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, n_max * n_max * 5 * sizeof(double2));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);
    cudaDeviceSynchronize();

    for (auto &n : n_list) {
        size_t m = n;

        for (auto &k : n_list) {
            double2 *cpuC      = work_cpu;
            double2 *cpuC1     = cpuC + m * n;
            double2 *cpuC2     = cpuC1 + m * n;
            double2 *devA      = reinterpret_cast<double2 *>(work_gpu);
            double2 *devB      = devA + m * k;
            double2 *devC      = devB + k * n;
            double2 *devC1     = devC + m * n;
            double2 *devC2     = devC1 + m * n;
            const size_t lda8i = ((k + 15) >> 4) << 4;
            const size_t ldb8i = lda8i;
            int8_t *A8i        = reinterpret_cast<int8_t *>(work_gemm);
            int8_t *B8i        = A8i + lda8i * m;
            int32_t *C32i      = reinterpret_cast<int32_t *>(B8i + ldb8i * n);
            double maxerr = 0.0, mederr = 0.0;

            //--------------------
            // generate matrices
            //--------------------
            makemat::randmat(m, k, devA, phi, seed);
            makemat::randmat(k, n, devB, phi, seed);

            //--------------------
            // C1+C2 := A*B (double-double arithmetic)
            //--------------------
            eval::dd_gpu::simple_gemm(m, n, k, devA, devB, devC1, devC2);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC1, devC1, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
            cudaMemcpy(cpuC2, devC2, m * n * sizeof(double2), cudaMemcpyDeviceToHost);

            //--------------------
            // C := A*B (int8-TC)
            //--------------------
            makemat::ones(lda8i * m + ldb8i * n, A8i);
            int32_t ialpha          = 1;
            int32_t ibeta           = 0;
            std::vector<double> res = getWatt::getWatt(
                [&]() {
                    cublasGemmEx(handle,
                                 CUBLAS_OP_T,
                                 CUBLAS_OP_N,
                                 m,
                                 n,
                                 lda8i,
                                 &ialpha,
                                 A8i,
                                 CUDA_R_8I,
                                 lda8i,
                                 B8i,
                                 CUDA_R_8I,
                                 ldb8i,
                                 &ibeta,
                                 C32i,
                                 CUDA_R_32I,
                                 m,
                                 CUBLAS_COMPUTE_32I,
                                 CUBLAS_GEMM_DEFAULT);
                },
                m,
                n,
                k);

            outFile << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
            outFile << "," << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
            std::cout << "," << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;

            //--------------------
            // C := A*B by FP64
            //--------------------
            double2 alpha{1.0, 0.0};
            double2 beta{0.0, 0.0};
            cudaDeviceSynchronize();
            res = getWatt::getWatt(
                [&]() {
                    cublasGemmEx(handle,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 m,
                                 n,
                                 k,
                                 &alpha,
                                 devA,
                                 CUDA_C_64F,
                                 m,
                                 devB,
                                 CUDA_C_64F,
                                 k,
                                 &beta,
                                 devC,
                                 CUDA_C_64F,
                                 m,
                                 CUBLAS_COMPUTE_64F,
                                 CUBLAS_GEMM_DEFAULT);
                },
                m,
                n,
                k);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "ZGEMMEx" << ",";
            outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "ZGEMMEx" << ",";
            std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;

            cudaDeviceSynchronize();
            res = getWatt::getWatt(
                [&]() {
                    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                &alpha, devA, m, devB, k,
                                &beta, devC, m);
                },
                m,
                n,
                k);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "ZGEMM" << ",";
            outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "ZGEMM" << ",";
            std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;

#if defined(__NVCC__)
            cudaDeviceSynchronize();
            res = getWatt::getWatt(
                [&]() {
                    cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                  &alpha, devA, m, devB, k,
                                  &beta, devC, m);
                },
                m,
                n,
                k);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "ZGEMM3m" << ",";
            outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "ZGEMM3m" << ",";
            std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
#endif

            //--------------------
            // C := A*B by ozaki-scheme2
            //--------------------
            for (auto &num_moduli : num_moduli_list) {
                cudaDeviceSynchronize();

                res = getWatt::getWatt(
                    [&]() {
                        gemmul8::gemm<double2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
                    },
                    m,
                    n,
                    k);

                cudaDeviceSynchronize();
                cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

                outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
                outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
                std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
                std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
            }

            //--------------------
            // C := A*B by ozaki-scheme2
            //--------------------
            for (auto &num_moduli : num_moduli_list) {
                cudaDeviceSynchronize();

                res = getWatt::getWatt(
                    [&]() {
                        gemmul8::gemm<double2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
                    },
                    m,
                    n,
                    k);

                cudaDeviceSynchronize();
                cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

                outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
                outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
                std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
                std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
            }
        }
    }

    delete[] work_cpu;
    cudaFree(work_gpu);
    cudaFree(work_gemm);
    cublasDestroy(handle);
    outFile.close();
}

void flops_ozaki1_check(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_z_time_ozaki1_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    outFile << std::scientific;
    std::cout << std::scientific;
    outFile << "phi,m,n,k,"
            << "function,"
            << "relerr_max,relerr_med,"
            << "TFLOPS,"
            << "total_time [sec],"
            << "conv_64f_2_8i,"
            << "cublasGemmEx,"
            << "conv_32i_2_8u,"
            << "inverse_scaling,"
            << std::endl;
    std::cout << "phi,m,n,k,"
              << "function,"
              << "relerr_max,relerr_med,"
              << "TFLOPS,"
              << "total_time [sec],"
              << "conv_64f_2_8i,"
              << "cublasGemmEx,"
              << "conv_32i_2_8u,"
              << "inverse_scaling,"
              << std::endl;
    cublasHandle_t handle;
    cublasCreate(&handle);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    const double phi        = 0.5;
    std::vector<size_t> n_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};
    const int itermax = AVERAGE;

    //--------------------
    // workspace
    //--------------------
    const size_t n_max          = *max_element(begin(n_list), end(n_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double2 *work_cpu           = new double2[n_max * n_max * 3];
    void *work_gpu;
    cudaMalloc(&work_gpu, n_max * n_max * sizeof(double2) * 4);
    cudaDeviceSynchronize();

#if defined(CUBLAS_VERSION_1300u2)
    cublasHandle_t handle_ozaki1;
    cublasCreate(&handle_ozaki1);
    cudaStream_t stream = NULL;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream(handle_ozaki1, stream);
    cudaEmulationMantissaControl_t mControl = CUDA_EMULATION_MANTISSA_CONTROL_FIXED;
    size_t workspaceSizeInBytes             = getFixedPointWorkspaceSizeInBytes(n_max, n_max, n_max, 1, true, mControl, 79);
    void *workspace;
    cudaMalloc(reinterpret_cast<void **>(&workspace), workspaceSizeInBytes);
    cublasSetWorkspace(handle_ozaki1, workspace, workspaceSizeInBytes);
    cublasSetMathMode(handle_ozaki1, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH);
    cublasSetEmulationStrategy(handle_ozaki1, CUBLAS_EMULATION_STRATEGY_EAGER);
    cublasSetFixedPointEmulationMantissaControl(handle_ozaki1, mControl);

    cudaDeviceSynchronize();

    for (auto &n : n_list) {
        size_t m       = n;
        size_t k       = n;
        double2 *cpuC  = work_cpu;
        double2 *cpuC1 = cpuC + m * n;
        double2 *cpuC2 = cpuC1 + m * n;
        double2 *devA  = reinterpret_cast<double2 *>(work_gpu);
        double2 *devB  = devA + m * k;
        double2 *devC  = devB + k * n;
        double2 *devC1 = devC;
        double2 *devC2 = devC1 + m * n;
        double maxerr = 0.0, mederr = 0.0;
        double time = 0.0;
        std::chrono::system_clock::time_point start, stop;

        //--------------------
        // generate matrices
        //--------------------
        makemat::randmat(m, k, devA, phi, seed);
        makemat::randmat(k, n, devB, phi, seed);

        //--------------------
        // C1+C2 := A*B (double-double arithmetic)
        //--------------------
        eval::dd_gpu::simple_gemm(m, n, k, devA, devB, devC1, devC2);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuC1, devC1, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuC2, devC2, m * n * sizeof(double2), cudaMemcpyDeviceToHost);

        double2 alpha{1.0, 0.0};
        double2 beta{0.0, 0.0};

        for (int mantissaBitCount = 55; mantissaBitCount < 80; mantissaBitCount += 8) {
            cublasSetFixedPointEmulationMaxMantissaBitCount(handle_ozaki1, mantissaBitCount);
            cudaDeviceSynchronize();
            cublasZgemm(handle_ozaki1, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                        &alpha, devA, m, devB, k,
                        &beta, devC, m);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start = std::chrono::system_clock::now();
                cublasZgemm(handle_ozaki1, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                            &alpha, devA, m, devB, k,
                            &beta, devC, m);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            }
            time = time / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "OS1-" << std::to_string((mantissaBitCount + 1) / 8) << ",";
            outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << "," << "," << "," << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS1-" << std::to_string((mantissaBitCount + 1) / 8) << ",";
            std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << "," << "," << "," << "," << std::endl;
        }
    }
#endif

    delete[] work_cpu;
    cudaFree(work_gpu);
    cublasDestroy(handle_ozaki1);
    cudaStreamDestroy(stream);
    cudaDeviceReset();
    outFile.close();
}

void exp_check(std::string &deviceName, std::string &dateTime) {
    std::cout << std::scientific;

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    std::vector<double> phi_list{PHI};
    std::vector<size_t> k_list{SIZE};
    const size_t m = 1024;
    const size_t n = 1024;

    //--------------------
    // workspace
    //--------------------
    const size_t k    = *max_element(begin(k_list), end(k_list));
    double2 *work_cpu = new double2[m * k + k * n];
    double2 *cpuA     = work_cpu;
    double2 *cpuB     = cpuA + m * k;
    void *work_gpu;
    cudaMalloc(&work_gpu, (m * k + k * n) * sizeof(double2));
    double2 *devA = reinterpret_cast<double2 *>(work_gpu);
    double2 *devB = devA + m * k;

    cudaDeviceSynchronize();
    std::cout << "phi,m,n,k,maxA,minA,medianA,q1A,q3A,maxB,minB,medianB,q1B,q3B," << std::endl;

    for (auto &phi : phi_list) {

        //--------------------
        // generate matrices
        //--------------------
        makemat::randmat(m, k, devA, phi, seed);
        makemat::randmat(k, n, devB, phi, seed);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuA, devA, m * k * sizeof(double2), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuB, devB, k * n * sizeof(double2), cudaMemcpyDeviceToHost);

        double maxA, minA, medA, q1A, q3A, maxB, minB, medB, q1B, q3B;
        eval::data_analysis(m, k, cpuA, maxA, minA, medA, q1A, q3A);
        eval::data_analysis(k, n, cpuB, maxB, minB, medB, q1B, q3B);
        std::cout << phi << ","
                  << m << ","
                  << n << ","
                  << k << ","
                  << maxA << ","
                  << minA << ","
                  << medA << ","
                  << q1A << ","
                  << q3A << ","
                  << maxB << ","
                  << minB << ","
                  << medB << ","
                  << q1B << ","
                  << q3B << ","
                  << std::endl;
    }

    delete[] work_cpu;
    cudaFree(work_gpu);
}

int main(int argc, char **argv) {
    std::chrono::system_clock::time_point start, stop;
    std::string deviceName = getDeviceName();
    std::string startTime  = getCurrentDateTime(start);

    bool run_accuracy     = false;
    bool run_flops        = false;
    bool run_watt         = false;
    bool run_flops_rect   = false;
    bool run_watt_rect    = false;
    bool run_flops_ozaki1 = false;
    bool run_exp_check    = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "accuracy") {
            run_accuracy = true;
        } else if (arg == "flops") {
            run_flops = true;
        } else if (arg == "watt") {
            run_watt = true;
        } else if (arg == "flops_rect") {
            run_flops_rect = true;
        } else if (arg == "watt_rect") {
            run_watt_rect = true;
        } else if (arg == "flops_ozaki1") {
            run_flops_ozaki1 = true;
        } else if (arg == "exp_check") {
            run_exp_check = true;
        } else if (arg == "all") {
            run_accuracy     = true;
            run_flops        = true;
            run_watt         = true;
            run_flops_rect   = true;
            run_watt_rect    = true;
            run_flops_ozaki1 = true;
            run_exp_check    = true;
        }
    }

    if (run_accuracy)
        accuracy_check(deviceName, startTime);
    if (run_flops)
        time_check(deviceName, startTime);
    if (run_watt)
        watt_check(deviceName, startTime);
    if (run_flops_rect)
        time_check_rect(deviceName, startTime);
    if (run_watt_rect)
        watt_check_rect(deviceName, startTime);
    if (run_flops_ozaki1)
        flops_ozaki1_check(deviceName, startTime);
    if (run_exp_check)
        exp_check(deviceName, startTime);

    std::string endTime = getCurrentDateTime(stop);
    auto sec            = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() * 1.e-9;
    std::cout << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "start        : " << startTime << std::endl;
    std::cout << "end          : " << endTime << std::endl;
    std::cout << "elapsed time : " << sec << " [sec]" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << std::endl;

    return 0;
}
