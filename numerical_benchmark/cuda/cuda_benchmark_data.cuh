#ifndef CUDA_BENCHMARK_DATA_H
#define CUDA_BENCHMARK_DATA_H

#include "../cpu/benchmark_data.hpp"
#include "../../common/cuda_err.cuh"
#include "../../common/helper_functions.hpp"
#include "transpose.cuh"

// adds cuda support to benchmark data
class cuBenchmarkData : public BenchmarkData
{
private:
    float *d_shift_data;
    float *d_rotate_data;
    float *d_rotate_data_transpose;
    int *d_shuffle_data;

public:
    cuBenchmarkData(int function, int dim);
    cuBenchmarkData(int function, int dim, int device);
    ~cuBenchmarkData();
    float *get_shift_ptr() { return d_shift_data; }
    float *get_rotate_ptr() { return d_rotate_data; }
    float *get_rotate_transpose_ptr() { return d_rotate_data_transpose; }
    int *get_shuffle_ptr() { return d_shuffle_data; }
};

cuBenchmarkData::cuBenchmarkData(int function, int dim) : BenchmarkData(function, dim)
{
    auto cf_num = 1;
    if (function >= composition_start) // composition functions
        cf_num = composition_functions.at(function);

    cuda_error_check(cudaMalloc((void **)&d_shift_data, dim * cf_num * sizeof(float)));
    cuda_error_check(cudaMalloc((void **)&d_rotate_data, dim * dim * cf_num * sizeof(float)));
    cuda_error_check(cudaMalloc((void **)&d_rotate_data_transpose, dim * dim * cf_num * sizeof(float)));
    cuda_error_check(cudaMalloc((void **)&d_shuffle_data, dim * sizeof(int)));

    cuda_error_check(cudaMemcpy(d_shift_data, shift_data, dim * cf_num * sizeof(float), cudaMemcpyHostToDevice));
    cuda_error_check(cudaMemcpy(d_rotate_data, rotate_data, dim * dim * cf_num * sizeof(float), cudaMemcpyHostToDevice));

    // transpose the rotation data so it is read contiguously in cuda kernel 
    dim3 grid(int_divide_up(dim, BLOCK_DIM), int_divide_up(dim, BLOCK_DIM));
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    for (int i = 0; i < cf_num; i++)
        transpose<<<grid, threads>>>(&d_rotate_data_transpose[i * dim * dim], &d_rotate_data[i * dim * dim], dim, dim);

    if (function >= hybrid_start && function <= composition_start)
        cuda_error_check(cudaMemcpy(d_shuffle_data, shuffle_data, dim * sizeof(int), cudaMemcpyHostToDevice));
}

cuBenchmarkData::cuBenchmarkData(int function, int dim, int device) : BenchmarkData(function, dim)
{
    cudaSetDevice(device); 
    auto cf_num = 1;
    if (function >= composition_start) // composition functions
        cf_num = composition_functions.at(function);

    cuda_error_check(cudaMalloc((void **)&d_shift_data, dim * cf_num * sizeof(float)));
    cuda_error_check(cudaMalloc((void **)&d_rotate_data, dim * dim * cf_num * sizeof(float)));
    cuda_error_check(cudaMalloc((void **)&d_rotate_data_transpose, dim * dim * cf_num * sizeof(float)));
    cuda_error_check(cudaMalloc((void **)&d_shuffle_data, dim * sizeof(int)));

    cuda_error_check(cudaMemcpy(d_shift_data, shift_data, dim * cf_num * sizeof(float), cudaMemcpyHostToDevice));
    cuda_error_check(cudaMemcpy(d_rotate_data, rotate_data, dim * dim * cf_num * sizeof(float), cudaMemcpyHostToDevice));

    // transpose the rotation data so it is read contiguously in cuda kernel 
    dim3 grid(int_divide_up(dim, BLOCK_DIM), int_divide_up(dim, BLOCK_DIM));
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    for (int i = 0; i < cf_num; i++)
        transpose<<<grid, threads>>>(&d_rotate_data_transpose[i * dim * dim], &d_rotate_data[i * dim * dim], dim, dim);

    if (function >= hybrid_start && function <= composition_start)
        cuda_error_check(cudaMemcpy(d_shuffle_data, shuffle_data, dim * sizeof(int), cudaMemcpyHostToDevice));
}

cuBenchmarkData::~cuBenchmarkData()
{
    cuda_error_check(cudaFree(d_shift_data));
    cuda_error_check(cudaFree(d_rotate_data));
    cuda_error_check(cudaFree(d_shuffle_data));
}

#endif