#ifndef CU_RANDOM_CUH
#define CU_RANDOM_CUH
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>

#include <stdio.h>

__device__ const float PI = 3.14159265358979323846;

namespace cg = cooperative_groups;

// CURAND initialisation
__global__ void initialise_curand_kernel(
    curandState *__restrict__ state,
    const unsigned long seed)
{
    const unsigned int tid = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    curand_init(seed, tid, 0, &state[tid]);
}

/*
 random number class for device side functions
 provides overloads for thread based version where each thread produces a random value
 and a thread group based version where thread 0 of the tg produces the random value and 
 uses shuffle instructions to sync it to the group 
*/
template <int tile_sz>
class cu_random
{
private:
    curandState *state;

public:
    __device__ cu_random(curandState *state) : state(state){};
    __device__ int random_int(const int lower, const int upper);
    __device__ int random_int(const cg::thread_block_tile<tile_sz> &g, const int lower, const int upper);
    template <typename... Ts>
    __device__ int mutually_exclusive_random_int(const int lower, const int upper, Ts... args);
    template <typename... Ts>
    __device__ int mutually_exclusive_random_int(const cg::thread_block_tile<tile_sz> &g, const int lower, const int upper, Ts... args);
    __device__ float uniform();
    __device__ float uniform(const cg::thread_block_tile<tile_sz> &g);
    __device__ float uniform(const float min, const float max);
    __device__ float uniform(const cg::thread_block_tile<tile_sz> &g, const float min, const float max);
    __device__ float cauchy(const float mu, const float gamma);
    __device__ float cauchy(const cg::thread_block_tile<tile_sz> &g, const float mu, const float gamma);
    __device__ float normal(const float mu, const float sigma);
    __device__ float normal(const cg::thread_block_tile<tile_sz> &g, const float mu, const float sigma);
};

template <int tile_sz>
__device__ int cu_random<tile_sz>::random_int(const int lower, const int upper)
{
    return lower + (curand(state) % (upper - lower));
}

template <int tile_sz>
__device__ int cu_random<tile_sz>::random_int(const cg::thread_block_tile<tile_sz> &g, const int lower, const int upper)
{
    int rnd;
    if (g.thread_rank() == 0)
    {
        rnd = lower + (curand(state) % (upper - lower));
    }
    g.sync();
    rnd = g.shfl(rnd, 0);
    return rnd;
}

template <int tile_sz>
template <typename... Ts>
__device__ int cu_random<tile_sz>::mutually_exclusive_random_int(const int lower, const int upper, Ts... args)
{
    int rnd;
    do
    {
        rnd = lower + (curand(state) % (upper - lower));
    } while (((rnd == args) || ...));
    return rnd;
}

template <int tile_sz>
template <typename... Ts>
__device__ int cu_random<tile_sz>::mutually_exclusive_random_int(const cg::thread_block_tile<tile_sz> &g, const int lower, const int upper, Ts... args)
{
    int rnd;
    if (g.thread_rank() == 0)
    {
        do
        {
            rnd = lower + (curand(state) % (upper - lower));
        } while (((rnd == args) || ...));
    }
    rnd = g.shfl(rnd, 0);
    return rnd;
}

template <int tile_sz>
__device__ float cu_random<tile_sz>::uniform()
{
    return curand_uniform(state);
}

template <int tile_sz>
__device__ float cu_random<tile_sz>::uniform(const cg::thread_block_tile<tile_sz> &g)
{
    float rnd;
    if (g.thread_rank() == 0)
        rnd = curand_uniform(state);
    rnd = g.shfl(rnd, 0);
    return rnd;
}

template <int tile_sz>
__device__ float cu_random<tile_sz>::uniform(const float min, const float max)
{
    return min + curand_uniform(state) * (max - min);
}

template <int tile_sz>
__device__ float cu_random<tile_sz>::uniform(const cg::thread_block_tile<tile_sz> &g, const float min, const float max)
{
    float rnd;
    if (g.thread_rank() == 0)
        rnd = min + curand_uniform(state) * (max - min);
    rnd = g.shfl(rnd, 0);
    return rnd;
}

template <int tile_sz>
__device__ float cu_random<tile_sz>::cauchy(const float mu, const float gamma)
{
    return mu + gamma * tanf(PI * (curand_uniform(state) - 0.5f));
}

template <int tile_sz>
__device__ float cu_random<tile_sz>::cauchy(const cg::thread_block_tile<tile_sz> &g, const float mu, const float gamma)
{
    float rnd;
    if (g.thread_rank() == 0)
        rnd = mu + gamma * tanf(PI * (curand_uniform(state) - 0.5f));
    rnd = g.shfl(rnd, 0);
    return rnd;
}

template <int tile_sz>
__device__ float cu_random<tile_sz>::normal(const float mu, const float sigma)
{
    return mu + sigma * curand_normal(state);
}

template <int tile_sz>
__device__ float cu_random<tile_sz>::normal(const cg::thread_block_tile<tile_sz> &g, const float mu, const float sigma)
{
    float rnd;
    if (g.thread_rank() == 0)
        rnd = mu + sigma * curand_normal(state);
    rnd = g.shfl(rnd, 0);
    return rnd;
}

// end random class
#endif