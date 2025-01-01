#ifndef WARP_HYBRID_FUNCTIONS_CUH 
#define WARP_HYBRID_FUNCTIONS_CUH 

#include <cooperative_groups.h> 
#include "util.cuh"

namespace benchmarks_warp 
{
namespace cg = cooperative_groups;

template <int tile_sz>
__device__ __forceinline__ void hybrid_function_1(
    const cg::thread_block_tile<tile_sz> &g,
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
    scale_shift_and_rotate<tile_sz>(g, x, y, z, dim, shift_data, rotate_data, 1.0f, shift_flag, rotate_flag);
    shuffle<tile_sz>(g, z, y, dim, shuffle_data);

    const float percentages[] = {0.3f, 0.3f, 0.4f};
    float s = 0.0f;

    int offset = 0;
    int size = ceilf(percentages[0] * dim);
    bent_cigar_function<tile_sz>(g, y + offset, y, z, fitness, size, shift_data, rotate_data, shuffle_data, false, false);
    if (g.thread_rank() == 0)
        s += *fitness;

    offset += size;
    size = ceilf(percentages[1] * dim);
    hgbat_function<tile_sz>(g, y + offset, y, z, fitness, size, shift_data, rotate_data, shuffle_data, false, false);
    if (g.thread_rank() == 0)
        s += *fitness;

    offset += size;
    size = dim - offset;
    rastrigin_function<tile_sz>(g, y + offset, y, z, fitness, size, shift_data, rotate_data, shuffle_data, false, false);
    if (g.thread_rank() == 0)
        *fitness += s; 
}


template <int tile_sz>
__device__ __forceinline__ void hybrid_function_2(
    const cg::thread_block_tile<tile_sz> &g,
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
    scale_shift_and_rotate<tile_sz>(g, x, y, z, dim, shift_data, rotate_data, 1.0f, shift_flag, rotate_flag);
    shuffle<tile_sz>(g, z, y, dim, shuffle_data);

    const float percentages[] = {0.2f, 0.2f, 0.3f, 0.3f};
    float s = 0.0f;

    int offset = 0;
    int size = ceilf(percentages[0] * dim);
    expanded_schaffer_F6_function<tile_sz>(g, y + offset, y, z, fitness, size, shift_data, rotate_data, shuffle_data, false, false);
    if (g.thread_rank() == 0)
        s += *fitness;

    offset += size;
    size = ceilf(percentages[1] * dim);
    hgbat_function<tile_sz>(g, y + offset, y, z, fitness, size, shift_data, rotate_data, shuffle_data, false, false);
    if (g.thread_rank() == 0)
        s += *fitness;

    offset += size;
    size = ceilf(percentages[2] * dim);
    rosenbrock_function<tile_sz>(g, y + offset, y, z, fitness, size, shift_data, rotate_data, shuffle_data, false, false);
    if (g.thread_rank() == 0)
        s += *fitness;

    offset += size;
    size = dim - offset;
    schwefel_function<tile_sz>(g, y + offset, y, z, fitness, size, shift_data, rotate_data, shuffle_data, false, false);
    if (g.thread_rank() == 0)
        *fitness += s;
}


template <int tile_sz>
__device__ __forceinline__ void hybrid_function_3(
    const cg::thread_block_tile<tile_sz> &g,
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
    scale_shift_and_rotate<tile_sz>(g, x, y, z, dim, shift_data, rotate_data, 1.0f, shift_flag, rotate_flag);
    shuffle<tile_sz>(g, z, y, dim, shuffle_data);
    const float percentages[] = {0.3f, 0.2f, 0.2f, 0.1f, 0.2f};
    float s = 0.0f;

    int offset = 0;
    int size = ceilf(percentages[0] * dim);
    katsuura_function<tile_sz>(g, y + offset, y, z, fitness, size, shift_data, rotate_data, shuffle_data, false, false);
    if (g.thread_rank() == 0)
        s += *fitness;

    offset += size;
    size = ceilf(percentages[1] * dim);
    happycat_function<tile_sz>(g, y + offset, y, z, fitness, size, shift_data, rotate_data, shuffle_data, false, false);
    if (g.thread_rank() == 0)
        s += *fitness;

    offset += size;
    size = ceilf(percentages[2] * dim);
    expanded_griewank_plus_rosenbrock_function<tile_sz>(g, y + offset, y, z, fitness, size, shift_data, rotate_data, shuffle_data, false, false);
    if (g.thread_rank() == 0)
        s += *fitness;

    offset += size;
    size = ceilf(percentages[3] * dim);
    schwefel_function<tile_sz>(g, y + offset, y, z, fitness, size, shift_data, rotate_data, shuffle_data, false, false);
    if (g.thread_rank() == 0)
        s += *fitness;

    offset += size;
    size = dim - offset;
    ackley_function<tile_sz>(g, y + offset, y, z, fitness, size, shift_data, rotate_data, shuffle_data, false, false);
    if (g.thread_rank() == 0)
        *fitness += s;
}


}

#endif 