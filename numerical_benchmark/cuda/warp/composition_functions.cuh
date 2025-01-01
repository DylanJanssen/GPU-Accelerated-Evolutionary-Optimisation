#ifndef WARP_COMPOSITION_FUNCTIONS_CUH 
#define WARP_COMPOSITION_FUNCTIONS_CUH 

#include <cooperative_groups.h> 
#include "util.cuh"

namespace benchmarks_warp 
{
namespace cg = cooperative_groups;



// performs the final composition calculation
template <int tile_sz>
__device__ __forceinline__ void composition_calculation(
    const cg::thread_block_tile<tile_sz> &g,
    float *__restrict__ x,
    float *__restrict__ y, // temporary memory (must be shared)
    float *__restrict__ z, // temporary memory (must be shared)
    float *__restrict__ fitness,
    const int dim,
    float *__restrict__ shift_data,
    const float *delta,    // controls coverage range
    const float *bias,     // defines which optima is global optima
    float *fitness_values, // cf_num fitness values (must be shared)
    const int cf_num)
{
    float temp;
    float sq_sum;
    float *weight = z;
    float *weight_sum = weight + cf_num; 
    if (g.thread_rank() == 0) 
        *weight_sum = 0.0f; 
    for (int i = 0; i < cf_num; i++)
    {
        sq_sum = 0.0f;
        for (int j = g.thread_rank(); j < dim; j += g.size())
        {
            temp = x[j] - shift_data[i * dim + j];
            sq_sum += temp * temp;
        }
        sq_sum = warp_reduce_sum<tile_sz>(g, sq_sum); // parallel reduction stored in thread 0
        if (g.thread_rank() == 0)
        {
            fitness_values[i] += bias[i]; 
            weight[i] = (1.0f / sqrtf(sq_sum)) * expf(-sq_sum / (2.0f * dim * delta[i] * delta[i]));
            *weight_sum += weight[i];
        }
    }
    g.sync(); 
    if (g.thread_rank() == 0) 
    {
        float s = 0.0f;
        for (int i = 0; i < cf_num; i++)
        {
            // printf("%f ", weight[i]); 
            s += weight[i] / *weight_sum * fitness_values[i];
        }
        // printf("%f\n", *weight_sum);
        *fitness = s;
    }
}

template <int tile_sz>
__device__ __forceinline__ void composition_function_1(
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
    // thread 0 must keep a record of fitness values before the final calculation uses shared memory 
    const int cf_num = 3; 
    const float delta[] = {10.0f, 20.0f, 30.0f};
    const float heights[] = {1.0f, 10.0f, 1.0f};
    const float bias[] = {0.0f, 100.0f, 200.0f};
    float fitness_values[3]; // this is shared in other verson 
    
    rastrigin_function<tile_sz>(g, x, y, z, &fitness_values[0], dim, shift_data, rotate_data, shuffle_data, shift_flag, rotate_flag); 
    griewank_function<tile_sz>(g, x, y, z, &fitness_values[1], dim, &shift_data[dim], &rotate_data[dim*dim], shuffle_data, shift_flag, rotate_flag); 
    schwefel_function<tile_sz>(g, x, y, z, &fitness_values[2], dim, &shift_data[2*dim], &rotate_data[2*dim*dim], shuffle_data, shift_flag, rotate_flag); 
    // seems to be faster to get thread 0 to do this, perhaps compiler isn't using 
    // shared memory for fitness values as only thread 0 is actually using them 
    if (g.thread_rank() == 0) 
        for (int i = 0; i < cf_num; i++)
            fitness_values[i] *= heights[i];
    
    // if (g.thread_rank() == 0) 
        // printf("%d %f %f %f\n", g.meta_group_rank(), fitness_values[0], fitness_values[1], fitness_values[2]); 
    composition_calculation<tile_sz>(g, x, y, z, fitness, dim, shift_data, delta, bias, fitness_values, cf_num);
    g.sync(); 
}

// composition function 2 comprised of ackley, ellipsoidal, griewank, rastrigin
template <int tile_sz>
__device__ __forceinline__ void high_conditional_elliptic_function(
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
    float s = 0.0f;
    for (int i = g.thread_rank(); i < dim; i += g.size())
        s += powf(10.0f, 6.0f * i / (dim - 1)) * z[i] * z[i]; 
    s = warp_reduce_sum<tile_sz>(g, s); // parallel reduction
    if (g.thread_rank() == 0)
        *fitness = s;
    g.sync(); 
}

template <int tile_sz>
__device__ __forceinline__ void composition_function_2(
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
    // thread 0 must keep a record of fitness values before the final calculation uses shared memory 
    const int cf_num = 4; 
    const float delta[] = {10.0f, 20.0f, 30.0f, 40.0f};
    const float heights[] = {10.0f, 1e-6f, 10.0f, 1.0f};
    const float bias[] = {0.0f, 100.0f, 200.0f, 300.0f};
    float fitness_values[4]; 
    
    ackley_function<tile_sz>(g, x, y, z, &fitness_values[0], dim, shift_data, rotate_data, shuffle_data, shift_flag, rotate_flag); 
    high_conditional_elliptic_function<tile_sz>(g, x, y, z, &fitness_values[1], dim, &shift_data[dim], &rotate_data[dim*dim], shuffle_data, shift_flag, rotate_flag); 
    griewank_function<tile_sz>(g, x, y, z, &fitness_values[2], dim, &shift_data[2*dim], &rotate_data[2*dim*dim], shuffle_data, shift_flag, rotate_flag); 
    rastrigin_function<tile_sz>(g, x, y, z, &fitness_values[3], dim, &shift_data[3*dim], &rotate_data[3*dim*dim], shuffle_data, shift_flag, rotate_flag); 
    // seems to be faster to get thread 0 to do this, perhaps compiler isn't using 
    // shared memory for fitness values as only thread 0 is actually using them 
    if (g.thread_rank() == 0) 
        for (int i = 0; i < cf_num; i++)
            fitness_values[i] *= heights[i];
    composition_calculation<tile_sz>(g, x, y, z, fitness, dim, shift_data, delta, bias, fitness_values, cf_num);
    g.sync(); 
}

// composition function 3 comprised of eschaffer6, schwefel, griewank, rosenbrock, rastrigin
template <int tile_sz>
__device__ __forceinline__ void composition_function_3(
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
    // thread 0 must keep a record of fitness values before the final calculation uses shared memory 
    const int cf_num = 5; 
    const float delta[] = {10.0f, 20.0f, 20.0f, 30.0f, 40.0f};
    const float heights[] = {0.0005f, 1.0f, 10.0f, 1.0f, 10.0f};
    const float bias[] = {0.0f, 100.0f, 200.0f, 300.0f, 400.0f};
    float fitness_values[5]; 
    
    expanded_schaffer_F6_function<tile_sz>(g, x, y, z, &fitness_values[0], dim, shift_data, rotate_data, shuffle_data, shift_flag, rotate_flag); 
    schwefel_function<tile_sz>(g, x, y, z, &fitness_values[1], dim, &shift_data[dim], &rotate_data[dim*dim], shuffle_data, shift_flag, rotate_flag); 
    griewank_function<tile_sz>(g, x, y, z, &fitness_values[2], dim, &shift_data[2*dim], &rotate_data[2*dim*dim], shuffle_data, shift_flag, rotate_flag); 
    rosenbrock_function<tile_sz>(g, x, y, z, &fitness_values[3], dim, &shift_data[3*dim], &rotate_data[3*dim*dim], shuffle_data, shift_flag, rotate_flag); 
    rastrigin_function<tile_sz>(g, x, y, z, &fitness_values[4], dim, &shift_data[4*dim], &rotate_data[4*dim*dim], shuffle_data, shift_flag, rotate_flag); 
    // seems to be faster to get thread 0 to do this, perhaps compiler isn't using 
    // shared memory for fitness values as only thread 0 is actually using them 
    if (g.thread_rank() == 0) 
        for (int i = 0; i < cf_num; i++)
            fitness_values[i] *= heights[i];
    composition_calculation<tile_sz>(g, x, y, z, fitness, dim, shift_data, delta, bias, fitness_values, cf_num);
    g.sync(); 
}


}

#endif 