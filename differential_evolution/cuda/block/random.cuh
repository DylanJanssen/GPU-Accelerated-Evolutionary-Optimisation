#ifndef RANDOM_CUH 
#define RANDOM_CUH 

#include <curand.h>
#include <curand_kernel.h>

template <typename... Ts>
__device__ __forceinline__
int mutually_exclusive_random_int(const int lower, const int upper, curandState *state, Ts... args)
{
    int rnd;
    do
    {
        rnd = lower + (curand(state) % (upper - lower));
    } while (((rnd == args) || ...));
    return rnd;
}

__global__ void initialise_curand_kernel(
    curandState *__restrict__ state,
    const unsigned long seed)
{
    const unsigned int tid = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    curand_init(seed, tid, 0, &state[tid]);
}

#endif 
