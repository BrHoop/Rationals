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
#define PHI        0.0, 0.5, 1, 1.5
#define NUM_MODULI 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
#if defined(GPU_MEM_MB) && GPU_MEM_MB >= 21000
    #define SIZE 1024, 2048, 4096, 8192, 16384
#else
    #define SIZE 1024, 2048, 4096, 8192
#endif

#if CUBLAS_VER_MAJOR > 12 || (CUBLAS_VER_MAJOR == 12 && CUBLAS_VER_MINOR >= 9)
    #define CUBLAS_VERSION_129
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

void accuracy_check(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_c_accuracy_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    outFile << std::scientific;
    std::cout << std::scientific;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    std::vector<float> phi_list{PHI};
    std::vector<size_t> k_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};
    const size_t m = 1024;
    const size_t n = 1024;

    //--------------------
    // workspace
    //--------------------
    const size_t k_max          = *max_element(begin(k_list), end(k_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double2 *workd_cpu          = new double2[m * n];
    float2 *workf_cpu           = new float2[m * n];
    size_t worksize             = gemmul8::workSize<true>(m, n, k_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, (m * k_max + k_max * n + m * n) * (sizeof(double2) + sizeof(float2)));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);
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
            double2 *cpuCd = workd_cpu;
            float2 *cpuCf  = workf_cpu;
            double2 *devAd = reinterpret_cast<double2 *>(work_gpu);
            double2 *devBd = devAd + m * k;
            double2 *devCd = devBd + k * n;
            float2 *devAf  = reinterpret_cast<float2 *>(devCd + m * n);
            float2 *devBf  = devAf + m * k;
            float2 *devCf  = devBf + k * n;
            double errmax, errmed;

            //--------------------
            // generate matrices
            //--------------------
            makemat::randmat(m, k, devAf, phi, seed);
            makemat::randmat(k, n, devBf, phi, seed);

            //--------------------
            // C1+C2 := A*B by FP64
            //--------------------
            double2 alpha{1.0, 0.0};
            double2 beta{0.0, 0.0};
            makemat::f2d(m, k, devAf, devAd);
            makemat::f2d(k, n, devBf, devBd);
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devAd, CUDA_C_64F, m, devBd, CUDA_C_64F, k, &beta, devCd, CUDA_C_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCd, devCd, m * n * sizeof(double2), cudaMemcpyDeviceToHost);

            //--------------------
            // C := A*B by FP32
            //--------------------
            float2 alphaf{1.0f, 0.0f};
            float2 betaf{0.0f, 0.0f};
            cudaDeviceSynchronize();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_C_32F, m, devBf, CUDA_C_32F, k, &betaf, devCf, CUDA_C_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, errmax, errmed);

            outFile << phi << ",CGEMMEx (k=" + std::to_string(k) + "),";
            std::cout << phi << ",CGEMMEx (k=" + std::to_string(k) + "),";
            for (int i = 0; i < num_moduli_list.size(); ++i) {
                outFile << errmax << ",";
                std::cout << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;

            cudaDeviceSynchronize();
            cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                        &alphaf, devAf, m, devBf, k,
                        &betaf, devCf, m);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, errmax, errmed);

            outFile << phi << ",CGEMM (k=" + std::to_string(k) + "),";
            std::cout << phi << ",CGEMM (k=" + std::to_string(k) + "),";
            for (int i = 0; i < num_moduli_list.size(); ++i) {
                outFile << errmax << ",";
                std::cout << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;

#if defined(__NVCC__)
            cudaDeviceSynchronize();
            cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                          &alphaf, devAf, m, devBf, k,
                          &betaf, devCf, m);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, errmax, errmed);

            outFile << phi << ",CGEMM3m (k=" + std::to_string(k) + "),";
            std::cout << phi << ",CGEMM3m (k=" + std::to_string(k) + "),";
            for (int i = 0; i < num_moduli_list.size(); ++i) {
                outFile << errmax << ",";
                std::cout << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;

            //--------------------
            // C := A*B by FP32 with CUBLAS_COMPUTE_32F_FAST_TF32
            //--------------------
            cudaDeviceSynchronize();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_C_32F, m, devBf, CUDA_C_32F, k, &betaf, devCf, CUDA_C_32F, m, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, errmax, errmed);

            outFile << phi << ",CGEMM-TF32 (k=" + std::to_string(k) + "),";
            std::cout << phi << ",CGEMM-TF32 (k=" + std::to_string(k) + "),";
            for (int i = 0; i < num_moduli_list.size(); ++i) {
                outFile << errmax << ",";
                std::cout << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;
#endif

            //--------------------
            // C := A*B by FP32 with CUBLAS_COMPUTE_32F_EMULATED_16BFX9
            //--------------------
#if defined(CUBLAS_VERSION_129)
            if (prop.major * 10 + prop.minor >= 100) {
                cudaDeviceSynchronize();
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_C_32F, m, devBf, CUDA_C_32F, k, &betaf, devCf, CUDA_C_32F, m, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUBLAS_GEMM_DEFAULT);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuCf, cpuCd, errmax, errmed);

                outFile << phi << ",CGEMM-BF16X9 (k=" + std::to_string(k) + "),";
                std::cout << phi << ",CGEMM-BF16X9 (k=" + std::to_string(k) + "),";
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
                timestmp = gemmul8::gemm<float2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, m, devBf, k, &betaf, devCf, m, num_moduli, true, work_gemm);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuCf, cpuCd, errmax, errmed);
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
                timestmp = gemmul8::gemm<float2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, m, devBf, k, &betaf, devCf, m, num_moduli, false, work_gemm);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuCf, cpuCd, errmax, errmed);
                outFile << errmax << ",";
                std::cout << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;
        }
    }

    delete[] workd_cpu;
    delete[] workf_cpu;
    cudaFree(work_gpu);
    cudaFree(work_gemm);
    cublasDestroy(handle);
    outFile.close();
}

void time_check(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_c_time_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    outFile << std::scientific;
    std::cout << std::scientific;

    outFile << "phi,m,n,k,"
            << "function,"
            << "relerr_max,relerr_med,"
            << "TFLOPS,"
            << "total_time [sec],"
            << "conv_32f_2_8i,"
            << "cublasGemmEx,"
            << "conv_32i_2_8u,"
            << "inverse_scaling,"
            << std::endl;
    std::cout << "phi,m,n,k,"
              << "function,"
              << "relerr_max,relerr_med,"
              << "TFLOPS,"
              << "total_time [sec],"
              << "conv_32f_2_8i,"
              << "cublasGemmEx,"
              << "conv_32i_2_8u,"
              << "inverse_scaling,"
              << std::endl;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    const float phi         = 0.5;
    std::vector<size_t> n_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};
    const int itermax = AVERAGE;

    //--------------------
    // workspace
    //--------------------
    const size_t n_max          = *max_element(begin(n_list), end(n_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double2 *workd_cpu          = new double2[n_max * n_max];
    float2 *workf_cpu           = new float2[n_max * n_max];
    size_t worksize             = gemmul8::workSize<true>(n_max, n_max, n_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, n_max * n_max * 3 * sizeof(float2));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);
    cudaDeviceSynchronize();

    for (auto &n : n_list) {
        size_t m           = n;
        size_t k           = n;
        double2 *cpuCd     = workd_cpu;
        float2 *cpuCf      = workf_cpu;
        float2 *devAf      = reinterpret_cast<float2 *>(work_gpu);
        float2 *devBf      = devAf + m * k;
        float2 *devCf      = devBf + k * n;
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
        makemat::randmat(m, k, devAf, phi, seed);
        makemat::randmat(k, n, devBf, phi, seed);

        //--------------------
        // C1+C2 := A*B by FP64
        //--------------------
        void *workd_gpu;
        cudaMalloc(&workd_gpu, (m * k + k * n + m * n) * sizeof(double2));
        double2 *devAd = reinterpret_cast<double2 *>(workd_gpu);
        double2 *devBd = devAd + m * k;
        double2 *devCd = devBd + k * n;
        makemat::f2d(m, k, devAf, devAd);
        makemat::f2d(k, n, devBf, devBd);

        double2 alpha{1.0, 0.0};
        double2 beta{0.0, 0.0};
        cudaDeviceSynchronize();
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devAd, CUDA_C_64F, m, devBd, CUDA_C_64F, k, &beta, devCd, CUDA_C_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCd, devCd, m * n * sizeof(double2), cudaMemcpyDeviceToHost);

        cudaFree(workd_gpu);

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
        // C := A*B by FP32
        //--------------------
        float2 alphaf{1.0f, 0.0f};
        float2 betaf{0.0f, 0.0f};
        cudaDeviceSynchronize();
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_C_32F, m, devBf, CUDA_C_32F, k, &betaf, devCf, CUDA_C_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

        time = 0.0;
        for (int iter = 0; iter < itermax; ++iter) {
            cudaDeviceSynchronize();
            start = std::chrono::system_clock::now();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_C_32F, m, devBf, CUDA_C_32F, k, &betaf, devCf, CUDA_C_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            stop = std::chrono::system_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }
        time = time / itermax * 1.e-9;

        outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMMEx" << ",";
        outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                << "," << "," << "," << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMMEx" << ",";
        std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                  << "," << "," << "," << "," << std::endl;

        cudaDeviceSynchronize();
        cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                    &alphaf, devAf, m, devBf, k,
                    &betaf, devCf, m);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

        time = 0.0;
        for (int iter = 0; iter < itermax; ++iter) {
            cudaDeviceSynchronize();
            start = std::chrono::system_clock::now();
            cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                        &alphaf, devAf, m, devBf, k,
                        &betaf, devCf, m);
            cudaDeviceSynchronize();
            stop = std::chrono::system_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }
        time = time / itermax * 1.e-9;

        outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMM" << ",";
        outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                << "," << "," << "," << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMM" << ",";
        std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                  << "," << "," << "," << "," << std::endl;

#if defined(__NVCC__)
        cudaDeviceSynchronize();
        cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                      &alphaf, devAf, m, devBf, k,
                      &betaf, devCf, m);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

        time = 0.0;
        for (int iter = 0; iter < itermax; ++iter) {
            cudaDeviceSynchronize();
            start = std::chrono::system_clock::now();
            cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                          &alphaf, devAf, m, devBf, k,
                          &betaf, devCf, m);
            cudaDeviceSynchronize();
            stop = std::chrono::system_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }
        time = time / itermax * 1.e-9;

        outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMM3m" << ",";
        outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                << "," << "," << "," << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMM3m" << ",";
        std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                  << "," << "," << "," << "," << std::endl;

        //--------------------
        // C := A*B by FP32 with CUBLAS_COMPUTE_32F_FAST_TF32
        //--------------------
        cudaDeviceSynchronize();
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_C_32F, m, devBf, CUDA_C_32F, k, &betaf, devCf, CUDA_C_32F, m, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

        time = 0.0;
        for (int iter = 0; iter < itermax; ++iter) {
            cudaDeviceSynchronize();
            start = std::chrono::system_clock::now();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_C_32F, m, devBf, CUDA_C_32F, k, &betaf, devCf, CUDA_C_32F, m, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            stop = std::chrono::system_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }
        time = time / itermax * 1.e-9;

        outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMM-TF32" << ",";
        outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                << "," << "," << "," << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMM-TF32" << ",";
        std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                  << "," << "," << "," << "," << std::endl;
#endif

        //--------------------
        // C := A*B by FP32 with CUBLAS_COMPUTE_32F_EMULATED_16BFX9
        //--------------------
#if defined(CUBLAS_VERSION_129)
        if (prop.major * 10 + prop.minor >= 100) {
            cudaDeviceSynchronize();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_C_32F, m, devBf, CUDA_C_32F, k, &betaf, devCf, CUDA_C_32F, m, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start = std::chrono::system_clock::now();
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_C_32F, m, devBf, CUDA_C_32F, k, &betaf, devCf, CUDA_C_32F, m, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUBLAS_GEMM_DEFAULT);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            }
            time = time / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMM-BF16X9" << ",";
            outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << "," << "," << "," << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMM-BF16X9" << ",";
            std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << "," << "," << "," << "," << std::endl;
        }
#endif

        //--------------------
        // C := A*B by ozaki-scheme2
        //--------------------
        for (auto &num_moduli : num_moduli_list) {
            std::vector<double> times(4, 0);
            std::vector<double> timestmp(4, 0);

            cudaDeviceSynchronize();
            timestmp = gemmul8::gemm<float2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, m, devBf, k, &betaf, devCf, m, num_moduli, true, work_gemm);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start    = std::chrono::system_clock::now();
                timestmp = gemmul8::gemm<float2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, m, devBf, k, &betaf, devCf, m, num_moduli, true, work_gemm);
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
            timestmp = gemmul8::gemm<float2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, m, devBf, k, &betaf, devCf, m, num_moduli, false, work_gemm);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start    = std::chrono::system_clock::now();
                timestmp = gemmul8::gemm<float2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, m, devBf, k, &betaf, devCf, m, num_moduli, false, work_gemm);
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

    delete[] workd_cpu;
    delete[] workf_cpu;
    cudaFree(work_gpu);
    cudaFree(work_gemm);
    cublasDestroy(handle);
    outFile.close();
}

void watt_check(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_c_watt_" + deviceName + "_" + dateTime + ".csv";
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

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    const float phi         = 0.5;
    std::vector<size_t> n_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};

    //--------------------
    // workspace
    //--------------------
    const size_t n_max          = *max_element(begin(n_list), end(n_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double2 *workd_cpu          = new double2[n_max * n_max];
    float2 *workf_cpu           = new float2[n_max * n_max];
    size_t worksize             = gemmul8::workSize<true>(n_max, n_max, n_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, n_max * n_max * 3 * sizeof(float2));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);
    cudaDeviceSynchronize();

    for (auto &n : n_list) {
        size_t m           = n;
        size_t k           = n;
        double2 *cpuCd     = workd_cpu;
        float2 *cpuCf      = workf_cpu;
        float2 *devAf      = reinterpret_cast<float2 *>(work_gpu);
        float2 *devBf      = devAf + m * k;
        float2 *devCf      = devBf + k * n;
        const size_t lda8i = ((k + 15) >> 4) << 4;
        const size_t ldb8i = lda8i;
        int8_t *A8i        = reinterpret_cast<int8_t *>(work_gemm);
        int8_t *B8i        = A8i + lda8i * m;
        int32_t *C32i      = reinterpret_cast<int32_t *>(B8i + ldb8i * n);
        double maxerr = 0.0, mederr = 0.0;

        //--------------------
        // generate matrices
        //--------------------
        makemat::randmat(m, k, devAf, phi, seed);
        makemat::randmat(k, n, devBf, phi, seed);

        //--------------------
        // C1+C2 := A*B by FP64
        //--------------------
        void *workd_gpu;
        cudaMalloc(&workd_gpu, (m * k + k * n + m * n) * sizeof(double2));
        double2 *devAd = reinterpret_cast<double2 *>(workd_gpu);
        double2 *devBd = devAd + m * k;
        double2 *devCd = devBd + k * n;
        makemat::f2d(m, k, devAf, devAd);
        makemat::f2d(k, n, devBf, devBd);

        double2 alpha{1.0, 0.0};
        double2 beta{0.0, 0.0};
        cudaDeviceSynchronize();
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devAd, CUDA_C_64F, m, devBd, CUDA_C_64F, k, &beta, devCd, CUDA_C_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCd, devCd, m * n * sizeof(double2), cudaMemcpyDeviceToHost);

        cudaFree(workd_gpu);

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
        // C := A*B by FP32
        //--------------------
        float2 alphaf{1.0f, 0.0f};
        float2 betaf{0.0f, 0.0f};
        res = getWatt::getWatt(
            [&]() {
                cublasGemmEx(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             m,
                             n,
                             k,
                             &alphaf,
                             devAf,
                             CUDA_C_32F,
                             m,
                             devBf,
                             CUDA_C_32F,
                             k,
                             &betaf,
                             devCf,
                             CUDA_C_32F,
                             m,
                             CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT);
            },
            m,
            n,
            k);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

        outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMMEx" << ",";
        outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMMEx" << ",";
        std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;

        res = getWatt::getWatt(
            [&]() {
                cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                            &alphaf, devAf, m, devBf, k,
                            &betaf, devCf, m);
            },
            m,
            n,
            k);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

        outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMM" << ",";
        outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMM" << ",";
        std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;

#if defined(__NVCC__)
        res = getWatt::getWatt(
            [&]() {
                cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                              &alphaf, devAf, m, devBf, k,
                              &betaf, devCf, m);
            },
            m,
            n,
            k);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

        outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMM3m" << ",";
        outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMM3m" << ",";
        std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;

        //--------------------
        // C := A*B by FP32 with CUBLAS_COMPUTE_32F_FAST_TF32
        //--------------------
        res = getWatt::getWatt(
            [&]() {
                cublasGemmEx(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             m,
                             n,
                             k,
                             &alphaf,
                             devAf,
                             CUDA_C_32F,
                             m,
                             devBf,
                             CUDA_C_32F,
                             k,
                             &betaf,
                             devCf,
                             CUDA_C_32F,
                             m,
                             CUBLAS_COMPUTE_32F_FAST_TF32,
                             CUBLAS_GEMM_DEFAULT);
            },
            m,
            n,
            k);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

        outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMM-TF32" << ",";
        outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMM-TF32" << ",";
        std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
#endif

        //--------------------
        // C := A*B by FP32 with CUBLAS_COMPUTE_32F_EMULATED_16BFX9
        //--------------------
#if defined(CUBLAS_VERSION_129)
        if (prop.major * 10 + prop.minor >= 100) {
            res = getWatt::getWatt(
                [&]() {
                    cublasGemmEx(handle,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 m,
                                 n,
                                 k,
                                 &alphaf,
                                 devAf,
                                 CUDA_C_32F,
                                 m,
                                 devBf,
                                 CUDA_C_32F,
                                 k,
                                 &betaf,
                                 devCf,
                                 CUDA_C_32F,
                                 m,
                                 CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
                                 CUBLAS_GEMM_DEFAULT);
                },
                m,
                n,
                k);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMM-BF16X9" << ",";
            outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMM-BF16X9" << ",";
            std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
        }
#endif

        //--------------------
        // C := A*B by ozaki-scheme2
        //--------------------
        for (auto &num_moduli : num_moduli_list) {

            cudaDeviceSynchronize();
            res = getWatt::getWatt(
                [&]() {
                    gemmul8::gemm<float2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, m, devBf, k, &betaf, devCf, m, num_moduli, true, work_gemm);
                },
                m,
                n,
                k);

            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

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
                    gemmul8::gemm<float2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, m, devBf, k, &betaf, devCf, m, num_moduli, false, work_gemm);
                },
                m,
                n,
                k);

            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
        }
    }

    delete[] workd_cpu;
    delete[] workf_cpu;
    cudaFree(work_gpu);
    cudaFree(work_gemm);
    cublasDestroy(handle);
    outFile.close();
}

void time_check_rect(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_c_time-rect_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    outFile << std::scientific;
    std::cout << std::scientific;

    outFile << "phi,m,n,k,"
            << "function,"
            << "relerr_max,relerr_med,"
            << "TFLOPS,"
            << "total_time [sec],"
            << "conv_32f_2_8i,"
            << "cublasGemmEx,"
            << "conv_32i_2_8u,"
            << "inverse_scaling,"
            << std::endl;
    std::cout << "phi,m,n,k,"
              << "function,"
              << "relerr_max,relerr_med,"
              << "TFLOPS,"
              << "total_time [sec],"
              << "conv_32f_2_8i,"
              << "cublasGemmEx,"
              << "conv_32i_2_8u,"
              << "inverse_scaling,"
              << std::endl;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    const float phi         = 0.5;
    std::vector<size_t> n_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};
    const int itermax = AVERAGE;

    //--------------------
    // workspace
    //--------------------
    const size_t n_max          = *max_element(begin(n_list), end(n_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double2 *workd_cpu          = new double2[n_max * n_max];
    float2 *workf_cpu           = new float2[n_max * n_max];
    size_t worksize             = gemmul8::workSize<true>(n_max, n_max, n_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, n_max * n_max * 3 * sizeof(float2));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);
    cudaDeviceSynchronize();

    for (auto &n : n_list) {
        size_t m = n;

        for (auto &k : n_list) {
            double2 *cpuCd     = workd_cpu;
            float2 *cpuCf      = workf_cpu;
            float2 *devAf      = reinterpret_cast<float2 *>(work_gpu);
            float2 *devBf      = devAf + m * k;
            float2 *devCf      = devBf + k * n;
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
            makemat::randmat(m, k, devAf, phi, seed);
            makemat::randmat(k, n, devBf, phi, seed);

            //--------------------
            // C1+C2 := A*B by FP64
            //--------------------
            void *workd_gpu;
            cudaMalloc(&workd_gpu, (m * k + k * n + m * n) * sizeof(double2));
            double2 *devAd = reinterpret_cast<double2 *>(workd_gpu);
            double2 *devBd = devAd + m * k;
            double2 *devCd = devBd + k * n;
            makemat::f2d(m, k, devAf, devAd);
            makemat::f2d(k, n, devBf, devBd);

            double2 alpha{1.0, 0.0};
            double2 beta{0.0, 0.0};
            cudaDeviceSynchronize();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devAd, CUDA_C_64F, m, devBd, CUDA_C_64F, k, &beta, devCd, CUDA_C_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCd, devCd, m * n * sizeof(double2), cudaMemcpyDeviceToHost);

            cudaFree(workd_gpu);

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
            // C := A*B by FP32
            //--------------------
            float2 alphaf{1.0f, 0.0f};
            float2 betaf{0.0f, 0.0f};
            cudaDeviceSynchronize();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_C_32F, m, devBf, CUDA_C_32F, k, &betaf, devCf, CUDA_C_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start = std::chrono::system_clock::now();
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_C_32F, m, devBf, CUDA_C_32F, k, &betaf, devCf, CUDA_C_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            }
            time = time / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMMEx" << ",";
            outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << "," << "," << "," << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMMEx" << ",";
            std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << "," << "," << "," << "," << std::endl;

            cudaDeviceSynchronize();
            cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                        &alphaf, devAf, m, devBf, k,
                        &betaf, devCf, m);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start = std::chrono::system_clock::now();
                cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                            &alphaf, devAf, m, devBf, k,
                            &betaf, devCf, m);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            }
            time = time / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMM" << ",";
            outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << "," << "," << "," << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMM" << ",";
            std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << "," << "," << "," << "," << std::endl;

#if defined(__NVCC__)
            cudaDeviceSynchronize();
            cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                          &alphaf, devAf, m, devBf, k,
                          &betaf, devCf, m);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start = std::chrono::system_clock::now();
                cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                              &alphaf, devAf, m, devBf, k,
                              &betaf, devCf, m);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            }
            time = time / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMM3m" << ",";
            outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << "," << "," << "," << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMM3m" << ",";
            std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << "," << "," << "," << "," << std::endl;

            //--------------------
            // C := A*B by FP32 with CUBLAS_COMPUTE_32F_FAST_TF32
            //--------------------
            cudaDeviceSynchronize();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_C_32F, m, devBf, CUDA_C_32F, k, &betaf, devCf, CUDA_C_32F, m, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start = std::chrono::system_clock::now();
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_C_32F, m, devBf, CUDA_C_32F, k, &betaf, devCf, CUDA_C_32F, m, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            }
            time = time / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMM-TF32" << ",";
            outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << "," << "," << "," << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMM-TF32" << ",";
            std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << "," << "," << "," << "," << std::endl;
#endif

            //--------------------
            // C := A*B by FP32 with CUBLAS_COMPUTE_32F_EMULATED_16BFX9
            //--------------------
#if defined(CUBLAS_VERSION_129)
            if (prop.major * 10 + prop.minor >= 100) {
                cudaDeviceSynchronize();
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_C_32F, m, devBf, CUDA_C_32F, k, &betaf, devCf, CUDA_C_32F, m, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUBLAS_GEMM_DEFAULT);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

                time = 0.0;
                for (int iter = 0; iter < itermax; ++iter) {
                    cudaDeviceSynchronize();
                    start = std::chrono::system_clock::now();
                    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_C_32F, m, devBf, CUDA_C_32F, k, &betaf, devCf, CUDA_C_32F, m, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUBLAS_GEMM_DEFAULT);
                    cudaDeviceSynchronize();
                    stop = std::chrono::system_clock::now();
                    time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                }
                time = time / itermax * 1.e-9;

                outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMM-BF16X9" << ",";
                outFile << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                        << "," << "," << "," << "," << std::endl;
                std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMM-BF16X9" << ",";
                std::cout << maxerr << "," << mederr << "," << 8.0 * m * n * k / time * 1.e-12 << "," << time << ","
                          << "," << "," << "," << "," << std::endl;
            }
#endif

            //--------------------
            // C := A*B by ozaki-scheme2
            //--------------------
            for (auto &num_moduli : num_moduli_list) {
                std::vector<double> times(4, 0);
                std::vector<double> timestmp(4, 0);

                cudaDeviceSynchronize();
                timestmp = gemmul8::gemm<float2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, m, devBf, k, &betaf, devCf, m, num_moduli, true, work_gemm);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

                time = 0.0;
                for (int iter = 0; iter < itermax; ++iter) {
                    cudaDeviceSynchronize();
                    start    = std::chrono::system_clock::now();
                    timestmp = gemmul8::gemm<float2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, m, devBf, k, &betaf, devCf, m, num_moduli, true, work_gemm);
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
                timestmp = gemmul8::gemm<float2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, m, devBf, k, &betaf, devCf, m, num_moduli, false, work_gemm);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

                time = 0.0;
                for (int iter = 0; iter < itermax; ++iter) {
                    cudaDeviceSynchronize();
                    start    = std::chrono::system_clock::now();
                    timestmp = gemmul8::gemm<float2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, m, devBf, k, &betaf, devCf, m, num_moduli, false, work_gemm);
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

    delete[] workd_cpu;
    delete[] workf_cpu;
    cudaFree(work_gpu);
    cudaFree(work_gemm);
    cublasDestroy(handle);
    outFile.close();
}

void watt_check_rect(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_c_watt-rect_" + deviceName + "_" + dateTime + ".csv";
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

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    const float phi         = 0.5;
    std::vector<size_t> n_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};

    //--------------------
    // workspace
    //--------------------
    const size_t n_max          = *max_element(begin(n_list), end(n_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double2 *workd_cpu          = new double2[n_max * n_max];
    float2 *workf_cpu           = new float2[n_max * n_max];
    size_t worksize             = gemmul8::workSize<true>(n_max, n_max, n_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, n_max * n_max * 3 * sizeof(float));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);
    cudaDeviceSynchronize();

    for (auto &n : n_list) {
        size_t m = n;

        for (auto &k : n_list) {
            double2 *cpuCd     = workd_cpu;
            float2 *cpuCf      = workf_cpu;
            float2 *devAf      = reinterpret_cast<float2 *>(work_gpu);
            float2 *devBf      = devAf + m * k;
            float2 *devCf      = devBf + k * n;
            const size_t lda8i = ((k + 15) >> 4) << 4;
            const size_t ldb8i = lda8i;
            int8_t *A8i        = reinterpret_cast<int8_t *>(work_gemm);
            int8_t *B8i        = A8i + lda8i * m;
            int32_t *C32i      = reinterpret_cast<int32_t *>(B8i + ldb8i * n);
            double maxerr = 0.0, mederr = 0.0;

            //--------------------
            // generate matrices
            //--------------------
            makemat::randmat(m, k, devAf, phi, seed);
            makemat::randmat(k, n, devBf, phi, seed);

            //--------------------
            // C1+C2 := A*B by FP64
            //--------------------
            void *workd_gpu;
            cudaMalloc(&workd_gpu, (m * k + k * n + m * n) * sizeof(double2));
            double2 *devAd = reinterpret_cast<double2 *>(workd_gpu);
            double2 *devBd = devAd + m * k;
            double2 *devCd = devBd + k * n;
            makemat::f2d(m, k, devAf, devAd);
            makemat::f2d(k, n, devBf, devBd);

            double2 alpha{1.0, 0.0};
            double2 beta{0.0, 0.0};
            cudaDeviceSynchronize();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devAd, CUDA_C_64F, m, devBd, CUDA_C_64F, k, &beta, devCd, CUDA_C_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCd, devCd, m * n * sizeof(double2), cudaMemcpyDeviceToHost);

            cudaFree(workd_gpu);

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
            // C := A*B by FP32
            //--------------------
            float2 alphaf{1.0f, 0.0f};
            float2 betaf{0.0f, 0.0f};
            res = getWatt::getWatt(
                [&]() {
                    cublasGemmEx(handle,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 m,
                                 n,
                                 k,
                                 &alphaf,
                                 devAf,
                                 CUDA_C_32F,
                                 m,
                                 devBf,
                                 CUDA_C_32F,
                                 k,
                                 &betaf,
                                 devCf,
                                 CUDA_C_32F,
                                 m,
                                 CUBLAS_COMPUTE_32F,
                                 CUBLAS_GEMM_DEFAULT);
                },
                m,
                n,
                k);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMMEx" << ",";
            outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMMEx" << ",";
            std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;

            res = getWatt::getWatt(
                [&]() {
                    cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                &alphaf, devAf, m, devBf, k,
                                &betaf, devCf, m);
                },
                m,
                n,
                k);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMM" << ",";
            outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMM" << ",";
            std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;

#if defined(__NVCC__)
            res = getWatt::getWatt(
                [&]() {
                    cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                  &alphaf, devAf, m, devBf, k,
                                  &betaf, devCf, m);
                },
                m,
                n,
                k);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMM3m" << ",";
            outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMM3m" << ",";
            std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;

            //--------------------
            // C := A*B by FP32 with CUBLAS_COMPUTE_32F_FAST_TF32
            //--------------------
            res = getWatt::getWatt(
                [&]() {
                    cublasGemmEx(handle,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 m,
                                 n,
                                 k,
                                 &alphaf,
                                 devAf,
                                 CUDA_C_32F,
                                 m,
                                 devBf,
                                 CUDA_C_32F,
                                 k,
                                 &betaf,
                                 devCf,
                                 CUDA_C_32F,
                                 m,
                                 CUBLAS_COMPUTE_32F_FAST_TF32,
                                 CUBLAS_GEMM_DEFAULT);
                },
                m,
                n,
                k);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMM-TF32" << ",";
            outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMM-TF32" << ",";
            std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
#endif

            //--------------------
            // C := A*B by FP32 with CUBLAS_COMPUTE_32F_EMULATED_16BFX9
            //--------------------
#if defined(CUBLAS_VERSION_129)
            if (prop.major * 10 + prop.minor >= 100) {
                res = getWatt::getWatt(
                    [&]() {
                        cublasGemmEx(handle,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     m,
                                     n,
                                     k,
                                     &alphaf,
                                     devAf,
                                     CUDA_C_32F,
                                     m,
                                     devBf,
                                     CUDA_C_32F,
                                     k,
                                     &betaf,
                                     devCf,
                                     CUDA_C_32F,
                                     m,
                                     CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
                                     CUBLAS_GEMM_DEFAULT);
                    },
                    m,
                    n,
                    k);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

                outFile << phi << "," << m << "," << n << "," << k << "," << "CGEMM-BF16X9" << ",";
                outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
                std::cout << phi << "," << m << "," << n << "," << k << "," << "CGEMM-BF16X9" << ",";
                std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
            }
#endif

            //--------------------
            // C := A*B by ozaki-scheme2
            //--------------------
            for (auto &num_moduli : num_moduli_list) {

                cudaDeviceSynchronize();
                res = getWatt::getWatt(
                    [&]() {
                        gemmul8::gemm<float2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, m, devBf, k, &betaf, devCf, m, num_moduli, true, work_gemm);
                    },
                    m,
                    n,
                    k);

                cudaDeviceSynchronize();
                cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

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
                        gemmul8::gemm<float2>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, m, devBf, k, &betaf, devCf, m, num_moduli, false, work_gemm);
                    },
                    m,
                    n,
                    k);

                cudaDeviceSynchronize();
                cudaMemcpy(cpuCf, devCf, m * n * sizeof(float2), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

                outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
                outFile << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
                std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
                std::cout << maxerr << "," << mederr << "," << res[0] << "," << 4.0 * res[1] * 1.e-9 << "," << std::endl;
            }
        }
    }

    delete[] workd_cpu;
    delete[] workf_cpu;
    cudaFree(work_gpu);
    cudaFree(work_gemm);
    cublasDestroy(handle);
    outFile.close();
}

int main(int argc, char **argv) {
    std::chrono::system_clock::time_point start, stop;
    std::string deviceName = getDeviceName();
    std::string startTime  = getCurrentDateTime(start);

    bool run_accuracy   = false;
    bool run_flops      = false;
    bool run_watt       = false;
    bool run_flops_rect = false;
    bool run_watt_rect  = false;
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
        } else if (arg == "all") {
            run_accuracy   = true;
            run_flops      = true;
            run_watt       = true;
            run_flops_rect = true;
            run_watt_rect  = true;
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
