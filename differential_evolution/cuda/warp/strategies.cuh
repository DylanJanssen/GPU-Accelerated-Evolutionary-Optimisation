#ifndef WARP_STRATEGIES_H
#define WARP_STRATEGIES_H
#include <cooperative_groups.h>
#include "cu_random.cuh"

namespace cg = cooperative_groups;

namespace strategies_warp
{

__device__ __forceinline__ 
float parent_correction(const float val, const float parent_val, const float lower_bound, const float upper_bound)
{
    if (val < lower_bound) 
        return (lower_bound + parent_val) / 2.0f; 
    else if (val > upper_bound)
        return (upper_bound + parent_val) / 2.0f; 
    else 
        return val; 
}


template <int tile_sz> 
__device__ __forceinline__ 
float random_correction(cu_random<tile_sz> &rnd, const float val, const float lower_bound, const float upper_bound)
{
    if (val < lower_bound || val > upper_bound)
        return rnd.uniform(lower_bound, upper_bound); 
    else 
        return val; 
}

/*
 DE/rand/1/bin
inputs: 
    g - warp thread block tile
    rnd - warp-based random number generated 
    population - entire population 
    offspring - this offspring 
    dim - problem dimension 
    popsize
    crossover rate 
    f - differential weighting factor 
    lower bound
    upper bound 
 */
template <int tile_sz>
__device__ void rand_one_binary(
    const cg::thread_block_tile<tile_sz> &g,
    cu_random<tile_sz> &rnd,
    float *__restrict__ population,
    float *__restrict__ offspring,
    const int dim,
    const int popsize,
    const float crossover_rate,
    const float f,
    const float lower_bound, 
    const float upper_bound)
{
    int a, b, c, j;
    int island_idx = g.meta_group_rank();
    a = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx);
    b = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx, a);
    c = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx, a, b);
    j = rnd.random_int(g, 0, dim);
    for (int i = g.thread_rank(); i < dim; i += g.size())
    {
        if (rnd.uniform() < crossover_rate || i == j)
        {
            offspring[i] = population[a * dim + i] + f * (population[b * dim + i] - population[c * dim + i]);
//            offspring[i] = random_correction(rnd, offspring[i], lower_bound, upper_bound); 
            offspring[i] = parent_correction(offspring[i], population[island_idx * dim + i], lower_bound, upper_bound); 
        }
        else
            offspring[i] = population[island_idx * dim + i];
    }
    g.sync(); 
}

// DE/rand/2/bin
template <int tile_sz>
__device__ void rand_two_binary(
    const cg::thread_block_tile<tile_sz> &g,
    cu_random<tile_sz> &rnd,
    float *__restrict__ population,
    float *__restrict__ offspring,
    const int dim,
    const int popsize,
    const float crossover_rate,
    const float f,
    const float lower_bound, 
    const float upper_bound)
{
    int a, b, c, d, e, j;
    int island_idx = g.meta_group_rank();
    a = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx);
    b = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx, a);
    c = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx, a, b);
    d = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx, a, b, c);
    e = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx, a, b, c, d);
    j = rnd.random_int(g, 0, dim);

    for (int i = g.thread_rank(); i < dim; i += g.size())
    {
        if (rnd.uniform() < crossover_rate || i == j)
        {
            offspring[i] = population[a * dim + i] + f * (population[b * dim + i] - population[c * dim + i]) + f * (population[d * dim + i] - population[e * dim + i]);
            offspring[i] = parent_correction(offspring[i], population[island_idx * dim + i], lower_bound, upper_bound); 
        }
        else
            offspring[i] = population[island_idx * dim + i];
    }
    g.sync(); 
}

// DE/best/1/bin
template <int tile_sz>
__device__ void best_one_binary(
    const cg::thread_block_tile<tile_sz> &g,
    cu_random<tile_sz> &rnd,
    float *__restrict__ population,
    float *__restrict__ offspring,
    int best, 
    const int dim,
    const int popsize,
    const float crossover_rate,
    const float f,
    const float lower_bound, 
    const float upper_bound)
{
    int a, b, j;
    int island_idx = g.meta_group_rank();
    a = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx);
    b = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx, a);
    j = rnd.random_int(g, 0, dim);

    for (int i = g.thread_rank(); i < dim; i += g.size())
    {
        if (rnd.uniform() < crossover_rate || i == j)
        {
            offspring[i] = population[best * dim + i] + f * (population[a * dim + i] - population[b * dim + i]);
            offspring[i] = parent_correction(offspring[i], population[island_idx * dim + i], lower_bound, upper_bound); 
        }
        else
            offspring[i] = population[island_idx * dim + i];
    }
    g.sync(); 
}

// DE/best/2/bin
template <int tile_sz>
__device__ void best_two_binary(
    const cg::thread_block_tile<tile_sz> &g,
    cu_random<tile_sz> &rnd,
    float *__restrict__ population,
    float *__restrict__ offspring,
    int best, 
    const int dim,
    const int popsize,
    const float crossover_rate,
    const float f,
    const float lower_bound, const float upper_bound)
{
    int a, b, c, d, j;
    int island_idx = g.meta_group_rank();
    a = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx);
    b = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx, a);
    c = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx, a, b);
    d = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx, a, b, c);
    j = rnd.random_int(g, 0, dim);

    for (int i = g.thread_rank(); i < dim; i += g.size())
    {
        if (rnd.uniform() < crossover_rate || i == j)
        {
            offspring[i] = population[best * dim + i] + f * (population[a * dim + i] - population[b * dim + i]) + f * (population[c * dim + i] - population[d * dim + i]);
            offspring[i] = parent_correction(offspring[i], population[island_idx * dim + i], lower_bound, upper_bound); 
        }
        else
            offspring[i] = population[island_idx * dim + i];
    }
    g.sync(); 
}


// DE/current-to-best/2/bin
template <int tile_sz>
__device__ void current_to_best_two_binary(
    const cg::thread_block_tile<tile_sz> &g,
    cu_random<tile_sz> &rnd,
    float *__restrict__ population,
    float *__restrict__ offspring,
    int best, 
    const int dim,
    const int popsize,
    const float crossover_rate,
    const float f,
    const float lower_bound, const float upper_bound)
{
    int a, b, j;
    int island_idx = g.meta_group_rank();
    a = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx);
    b = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx, a);
    j = rnd.random_int(g, 0, dim);

    for (int i = g.thread_rank(); i < dim; i += g.size())
    {
        if (rnd.uniform() < crossover_rate || i == j)
        {
            offspring[i] = population[island_idx * dim + i] + f * (population[best * dim + i] - population[island_idx * dim + i]) + f * (population[a * dim + i] - population[b * dim + i]);
            offspring[i] = parent_correction(offspring[i], population[island_idx * dim + i], lower_bound, upper_bound); 
        }
        else
            offspring[i] = population[island_idx * dim + i];
    }
    g.sync(); 
}


// DE/rand-to-best/2/bin
template <int tile_sz>
__device__ void rand_to_best_two_binary(
    const cg::thread_block_tile<tile_sz> &g,
    cu_random<tile_sz> &rnd,
    float *__restrict__ population,
    float *__restrict__ offspring,
    int best,
    const int dim,
    const int popsize,
    const float crossover_rate,
    const float f,
    const float lower_bound, const float upper_bound)
{
    int a, b, c, d, j;
    int island_idx = g.meta_group_rank();
    a = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx);
    b = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx, a);
    c = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx, a, b);
    d = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx, a, b, c);
    j = rnd.random_int(g, 0, dim);

    for (int i = g.thread_rank(); i < dim; i += g.size())
    {
        if (rnd.uniform() < crossover_rate || i == j)
        {
            offspring[i] = population[island_idx * dim + i] + f * (population[best * dim + i] - population[island_idx * dim + i]) + f * (population[a * dim + i] - population[b * dim + i]) + f * (population[c * dim + i] - population[d * dim + i]);
            offspring[i] = parent_correction(offspring[i], population[island_idx * dim + i], lower_bound, upper_bound); 
        }
        else
            offspring[i] = population[island_idx * dim + i];
    }
    g.sync(); 
}

// DE/current-to-rand/1
template <int tile_sz>
__device__ void current_to_rand_one(
    const cg::thread_block_tile<tile_sz> &g,
    cu_random<tile_sz> &rnd,
    float *__restrict__ population,
    float *__restrict__ offspring,
    int best,
    const int dim,
    const int popsize,
    const float crossover_rate,
    const float f,
    const float lower_bound, const float upper_bound)
{
    int a, b, c, k;
    int island_idx = g.meta_group_rank();
    a = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx);
    b = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx, a);
    c = rnd.mutually_exclusive_random_int(g, 0, popsize, island_idx, a, b);
    k = rnd.uniform(g);
    for (int i = g.thread_rank(); i < dim; i += g.size())
    {
        offspring[i] = population[island_idx * dim + i] + k * (population[a * dim + i] - population[island_idx * dim + i]) + f * (population[b * dim + i] - population[c * dim + i]);
        offspring[i] = parent_correction(offspring[i], population[island_idx * dim + i], lower_bound, upper_bound); 
    }
    g.sync(); 
}
}

#endif