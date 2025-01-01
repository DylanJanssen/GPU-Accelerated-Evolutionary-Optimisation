#ifndef CUDA_BENCHMARKS_WARP_H
#define CUDA_BENCHMARKS_WARP_H

#include <cooperative_groups.h>

#include "../../common/cuda_err.cuh"
#include "cuda_benchmark_data.cuh"
#include "warp/basic_functions.cuh"
#include "warp/hybrid_functions.cuh"
#include "warp/composition_functions.cuh"


namespace benchmarks_warp
{

// driving function
template <int tile_sz> 
__device__
void evaluate(
    const cg::thread_block_tile<tile_sz> &g,
    int function_number,
    float *__restrict__ x,
    float *__restrict__ y, 
    float *__restrict__ z,
    float *__restrict__ fitness,
    const int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data)
{
    switch (function_number)
    {
    case 1:
        zakharov_function<tile_sz>(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 2:
        rosenbrock_function<tile_sz>(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 3:
        rastrigin_function<tile_sz>(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 4:
        schwefel_function<tile_sz>(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 5:
        hybrid_function_1<tile_sz>(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 6:
        hybrid_function_2<tile_sz>(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 7:
        hybrid_function_3<tile_sz>(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 8:
        composition_function_1<tile_sz>(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 9:
        composition_function_2<tile_sz>(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    case 10:
        composition_function_3<tile_sz>(g, x, y, z, fitness, dim, shift_data, rotate_data, shuffle_data, true, true);
        break;
    }
}
}

#endif 