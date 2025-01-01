#ifndef CUDA_BENCHMARKS_H
#define CUDA_BENCHMARKS_H

#include <cooperative_groups.h>

#include "../../common/cuda_err.cuh"
#include "cuda_benchmark_data.cuh"
#include "block/basic_functions.cuh"
#include "block/hybrid_functions.cuh"
#include "block/composition_functions.cuh"

namespace cg = cooperative_groups;


// GPU kernels
__global__ void zakharov_kernel(
    float *__restrict__ x,
    float *__restrict__ fitness,
    const int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    const bool shift_flag,
    const bool rotate_flag)
{
    extern __shared__ char shmem[];
    float *y = (float *)shmem;
    float *z = &y[next_power_of_two(dim)];
    cg::thread_block g = cg::this_thread_block();
    float *this_x = x + blockIdx.x * dim;
    float *this_fitness = fitness + blockIdx.x;
    benchmarks_block::zakharov_function(g, this_x, y, z, this_fitness, dim, shift_data, rotate_data, shuffle_data, shift_flag, rotate_flag);
}

__global__ void rosenbrock_kernel(
    float *__restrict__ x,
    float *__restrict__ fitness,
    const int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    const bool shift_flag,
    const bool rotate_flag)
{
    extern __shared__ char shmem[];
    float *y = (float *)shmem;
    float *z = &y[next_power_of_two(dim)];
    cg::thread_block g = cg::this_thread_block();
    float *this_x = x + blockIdx.x * dim;
    float *this_fitness = fitness + blockIdx.x;
    benchmarks_block::rosenbrock_function(g, this_x, y, z, this_fitness, dim, shift_data, rotate_data, shuffle_data, shift_flag, rotate_flag);
}

__global__ void rastrigin_kernel(
    float *__restrict__ x,
    float *__restrict__ fitness,
    const int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    const bool shift_flag,
    const bool rotate_flag)
{
    extern __shared__ char shmem[];
    float *y = (float *)shmem;
    float *z = &y[next_power_of_two(dim)];
    cg::thread_block g = cg::this_thread_block();
    float *this_x = x + blockIdx.x * dim;
    float *this_fitness = fitness + blockIdx.x;
    benchmarks_block::rastrigin_function(g, this_x, y, z, this_fitness, dim, shift_data, rotate_data, shuffle_data, shift_flag, rotate_flag);
}

__global__ void schwefel_kernel(
    float *__restrict__ x,
    float *__restrict__ fitness,
    const int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    const bool shift_flag,
    const bool rotate_flag)
{
    extern __shared__ char shmem[];
    float *y = (float *)shmem;
    float *z = &y[next_power_of_two(dim)];
    cg::thread_block g = cg::this_thread_block();
    float *this_x = x + blockIdx.x * dim;
    float *this_fitness = fitness + blockIdx.x;
    benchmarks_block::schwefel_function(g, this_x, y, z, this_fitness, dim, shift_data, rotate_data, shuffle_data, shift_flag, rotate_flag);
}

__global__ void hybrid_function_1_kernel(
    float *__restrict__ x,
    float *__restrict__ fitness,
    const int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    const bool shift_flag,
    const bool rotate_flag)
{
    extern __shared__ char shmem[];
    float *y = (float *)shmem;
    float *z = &y[next_power_of_two(dim)];
    cg::thread_block g = cg::this_thread_block();
    float *this_x = x + blockIdx.x * dim;
    float *this_fitness = fitness + blockIdx.x;
    benchmarks_block::hybrid_function_1(g, this_x, y, z, this_fitness, dim, shift_data, rotate_data, shuffle_data, shift_flag, rotate_flag);
}

__global__ void hybrid_function_2_kernel(
    float *__restrict__ x,
    float *__restrict__ fitness,
    const int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    const bool shift_flag,
    const bool rotate_flag)
{
    extern __shared__ char shmem[];
    float *y = (float *)shmem;
    float *z = &y[next_power_of_two(dim)];
    cg::thread_block g = cg::this_thread_block();
    float *this_x = x + blockIdx.x * dim;
    float *this_fitness = fitness + blockIdx.x;
    benchmarks_block::hybrid_function_2(g, this_x, y, z, this_fitness, dim, shift_data, rotate_data, shuffle_data, shift_flag, rotate_flag);
}

__global__ void hybrid_function_3_kernel(
    float *__restrict__ x,
    float *__restrict__ fitness,
    const int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    const bool shift_flag,
    const bool rotate_flag)
{
    extern __shared__ char shmem[];
    float *y = (float *)shmem;
    float *z = &y[next_power_of_two(dim)];
    cg::thread_block g = cg::this_thread_block();
    float *this_x = x + blockIdx.x * dim;
    float *this_fitness = fitness + blockIdx.x;
    benchmarks_block::hybrid_function_3(g, this_x, y, z, this_fitness, dim, shift_data, rotate_data, shuffle_data, shift_flag, rotate_flag);
}

__global__ void composition_function_1_kernel(
    float *__restrict__ x,
    float *__restrict__ fitness,
    const int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    const bool shift_flag,
    const bool rotate_flag)
{
    extern __shared__ char shmem[];
    float *y = (float *)shmem;
    float *z = &y[next_power_of_two(dim)];
    cg::thread_block g = cg::this_thread_block();
    float *this_x = x + blockIdx.x * dim;
    float *this_fitness = fitness + blockIdx.x;
    benchmarks_block::composition_function_1(g, this_x, y, z, this_fitness, dim, shift_data, rotate_data, shuffle_data, shift_flag, rotate_flag);
}

__global__ void composition_function_2_kernel(
    float *__restrict__ x,
    float *__restrict__ fitness,
    const int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    const bool shift_flag,
    const bool rotate_flag)
{
    extern __shared__ char shmem[];
    float *y = (float *)shmem;
    float *z = &y[next_power_of_two(dim)];
    cg::thread_block g = cg::this_thread_block();
    float *this_x = x + blockIdx.x * dim;
    float *this_fitness = fitness + blockIdx.x;
    benchmarks_block::composition_function_2(g, this_x, y, z, this_fitness, dim, shift_data, rotate_data, shuffle_data, shift_flag, rotate_flag);
}

__global__ void composition_function_3_kernel(
    float *__restrict__ x,
    float *__restrict__ fitness,
    const int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    const bool shift_flag,
    const bool rotate_flag)
{
    extern __shared__ char shmem[];
    float *y = (float *)shmem;
    float *z = &y[next_power_of_two(dim)];
    cg::thread_block g = cg::this_thread_block();
    float *this_x = x + blockIdx.x * dim;
    float *this_fitness = fitness + blockIdx.x;
    benchmarks_block::composition_function_3(g, this_x, y, z, this_fitness, dim, shift_data, rotate_data, shuffle_data, shift_flag, rotate_flag);
}

// driving function
void evaluate(
    int function_number,
    float *__restrict__ x,
    float *__restrict__ fitness,
    const int popsize,
    const int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data)
{
    int shared_bytes = next_power_of_two(dim) * 2 * sizeof(float);
    int block_size = next_power_of_two(dim);
    switch (function_number)
    {
    case 1:
        zakharov_kernel<<<popsize, block_size, shared_bytes>>>(x, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 2:
        rosenbrock_kernel<<<popsize, block_size, shared_bytes>>>(x, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 3:
        rastrigin_kernel<<<popsize, block_size, shared_bytes>>>(x, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 4:
        schwefel_kernel<<<popsize, block_size, shared_bytes>>>(x, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 5:
        hybrid_function_1_kernel<<<popsize, block_size, shared_bytes>>>(x, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 6:
        hybrid_function_2_kernel<<<popsize, block_size, shared_bytes>>>(x, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 7:
        hybrid_function_3_kernel<<<popsize, block_size, shared_bytes>>>(x, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 8:
        composition_function_1_kernel<<<popsize, block_size, shared_bytes>>>(x, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 9:
        composition_function_2_kernel<<<popsize, block_size, shared_bytes>>>(x, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 10:
        composition_function_3_kernel<<<popsize, block_size, shared_bytes>>>(x, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    }
    
    cuda_error_check(cudaPeekAtLastError());
    cuda_error_check(cudaDeviceSynchronize());
}

#endif 