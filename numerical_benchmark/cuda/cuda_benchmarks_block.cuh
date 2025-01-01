#ifndef CUDA_BENCHMARKS_H
#define CUDA_BENCHMARKS_H

#include <cooperative_groups.h>

#include "../../common/cuda_err.cuh"
#include "cuda_benchmark_data.cuh"
#include "block/basic_functions.cuh"
#include "block/hybrid_functions.cuh"
#include "block/composition_functions.cuh"

namespace cg = cooperative_groups;


__device__ __forceinline__
void function_evaluation(
    const cg::thread_block &g,
    int function_number,
    float *__restrict__ x,
    float *__restrict__ y,
    float *__restrict__ z,
    float *__restrict__ fitness,
    int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    bool shift_flag,
    bool rotate_flag)
{
    switch (function_number)
    {
    case 1:
        benchmarks_block::zakharov_function(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 2:
        benchmarks_block::rosenbrock_function(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 3:
        benchmarks_block::rastrigin_function(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 4:
        benchmarks_block::schwefel_function(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 5:
        benchmarks_block::hybrid_function_1(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 6:
        benchmarks_block::hybrid_function_2(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 7:
        benchmarks_block::hybrid_function_3(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 8:
        benchmarks_block::composition_function_1(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 9:
        benchmarks_block::composition_function_2(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 10:
        benchmarks_block::composition_function_3(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    }
    g.sync();
}

#endif 