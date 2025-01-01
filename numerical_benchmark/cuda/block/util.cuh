#ifndef BLOCK_UTIL_CUH 
#define BLOCK_UTIL_CUH 

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// helper functions
__host__ __device__ __forceinline__ int next_power_of_two(int x)
{
    int power = 1;
    while (power < x)
        power *= 2;
    return power;
}

__device__ __forceinline__ void shuffle(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ x_shuffle,
    const int dim,
    int *__restrict__ shuffle_data)
{
    for (int i = g.thread_rank(); i < dim; i += g.size())
        x_shuffle[i] = x[shuffle_data[i] - 1];
    g.sync();
}

__device__ __forceinline__ void scale(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ x_scale,
    const int dim,
    const float scale_rate)
{
    for (int i = g.thread_rank(); i < dim; i += g.size())
        x_scale[i] = x[i] * scale_rate;
    g.sync();
}

__device__ __forceinline__ void shift(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ x_shift,
    const int dim,
    float *__restrict__ shift_data)
{
    for (int i = g.thread_rank(); i < dim; i += g.size())
        x_shift[i] = x[i] - shift_data[i];
    g.sync();
}


// this rotation function assumes the rotation matrix is transposed 
__device__ __forceinline__ void rotate(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ x_rotate,
    const int dim,
    float *__restrict__ rotate_data_T)
{
    for (int i = g.thread_rank(); i < dim; i += g.size())
    {
        x_rotate[i] = 0.0f;
        for (int j = 0; j < dim; j++)
            x_rotate[i] += x[j] * rotate_data_T[i + dim * j];
    }
    g.sync();
}

// scales x by given scale rate
// shifts x by given scale data
// rotates x by given rotate data
// result is returned in z
// extra memory y is required for shift + rotate case
__device__ __forceinline__ void scale_shift_and_rotate(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ y,
    float *__restrict__ z,
    const int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data_T,
    const float scale_rate,
    const bool shift_flag,
    const bool rotate_flag)
{
    if (shift_flag)
    {
        shift(g, x, y, dim, shift_data); // shift x and store in y
        if (rotate_flag)
        {
            scale(g, y, y, dim, scale_rate);   // scale y and store in y
            rotate(g, y, z, dim, rotate_data_T); // rotate y and store in z
        }
        else
            scale(g, y, z, dim, scale_rate); // scale y and store in z
    }
    else if (rotate_flag)
    {
        scale(g, x, y, dim, scale_rate);   // scale x and store in y
        rotate(g, y, z, dim, rotate_data_T); // rotate y and store in z
    }
    else
        scale(g, x, z, dim, scale_rate); // scale x and store in z
}

// parallel reduction using temp array, ideally shared memory
// requires thread block to be a power of two threads.
__device__ __forceinline__ float reduce_sum(
    const cg::thread_block &g,
    float *__restrict__ temp,
    const float value,
    const int size)
{
    int lane = g.thread_rank();
    if (lane < size) // ensure that any sized array can be summed
        temp[lane] = value;
    g.sync();
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        if (lane < i && lane+i < size)
            temp[lane] += temp[lane + i];
        g.sync();
    }
    g.sync(); 
    return temp[lane];
}

// parallel reduction using temp array, ideally shared memory
// requires thread block to be a power of two threads.
__device__ __forceinline__ float reduce_product(
    const cg::thread_block &g,
    float *__restrict__ temp,
    const float value,
    const int size)
{
    int lane = g.thread_rank();
    if (lane < size) // ensure that any sized array can be summed
        temp[lane] = value;
    g.sync();
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        if (lane < i && lane+i < size)
            temp[lane] *= temp[lane + i];
        g.sync();
    }
    g.sync(); 
    return temp[lane];
}


#endif 