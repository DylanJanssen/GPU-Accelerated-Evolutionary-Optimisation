#include <cooperative_groups.h>
#include <iostream>
#include <algorithm>

#include "cuda/warp/cu_random.cuh"
#include "../numerical_benchmark/cuda/cuda_benchmarks_warp.cuh"
#include "cuda/warp/strategies.cuh"
#include "../common/cuvector.cuh"
#include "../common/cuda_err.cuh"
#include "../common/helper_functions.hpp"
#include "cuda/solution.cuh"
#include "cuda/warp/selection.cuh"

namespace cg = cooperative_groups;

__host__ __device__ __forceinline__ int log2_ceil(int value)
{
    int log2_value = 0;
    while ((1 << log2_value) < value)
        ++log2_value;
    return log2_value;
}

__host__ __device__ __forceinline__ int next_power_of_two(int x)
{
    int log2_elements = log2_ceil(x);
    return 1 << log2_elements;
}

template <int tile_sz>
__device__ __forceinline__ void initialise(
    const cg::thread_block_tile<tile_sz> &g,
    cu_random<tile_sz> &rnd,
    float *__restrict__ x,
    const int dim,
    const float lower_bound,
    const float upper_bound)
{
    for (int i = g.thread_rank(); i < dim; i += g.size())
        x[i] = rnd.uniform(lower_bound, upper_bound);
    g.sync();
}

template <int tile_sz>
__device__ __forceinline__ void replacement(
    const cg::thread_block_tile<tile_sz> &g,
    float *__restrict__ x,
    float *__restrict__ fitness,
    float *__restrict__ offspring,
    float *__restrict__ offspring_fitness,
    const int dim,
    solution *sol)
{
    if (*offspring_fitness < *fitness)
    {
        for (int i = g.thread_rank(); i < dim; i += g.size())
            x[i] = offspring[i];
        if (g.thread_rank() == 0)
        {
            *fitness = *offspring_fitness;
            check_solution(fitness, sol, g.meta_group_rank());
        }
    }
    g.sync();
}

/***************************
 * DIFFERENTIAL EVOLUTION  *
 ****************************/
template <int tile_sz>
__global__ void differential_evolution_kernel(
    int function,
    float *__restrict__ rotate,
    float *__restrict__ shift,
    int *__restrict__ shuffle,
    float *__restrict__ population,
    float *__restrict__ fitness,
    float *__restrict__ migrants,
    float *__restrict__ migrant_fitness,
    const float F,
    const float CR,
    const int num_migrants,
    const int dim,
    const int popsize,
    const float lower_bound,
    const float upper_bound,
    const int evaluations,
    const int migration,
    curandState *__restrict__ states,
    solution *sol)
{
    const auto thread_block = cg::this_thread_block();
    const auto g = cg::tiled_partition<tile_sz>(thread_block); // splits thread_block into thread block tiles of tile_sz
    const int block_tid = thread_block.thread_rank();          // overall thread number
    const int island_idx = g.meta_group_rank();                // population individual number
    if (island_idx >= popsize)                                 // any extra threads return
        return;

    // population and fitness values in global memory
    float *island_population = &population[blockIdx.x * popsize * dim];
    float *island_fitness = &fitness[blockIdx.x * popsize];

    // setup shared memory
    extern __shared__ char shmem[];
    int memory_size = next_power_of_two(dim);
    float *y = (float *)shmem;                      // dynamically allocated shared memory
    float *this_y = &y[island_idx * memory_size];   // used for objective functions
    float *this_z = &this_y[popsize * memory_size]; // used for objective functions
    float *this_offspring = &this_z[popsize * memory_size];
    float *this_offspring_fitness = &y[3 * popsize * memory_size + island_idx];

    cu_random<tile_sz> rnd(&states[blockIdx.x * blockDim.x + block_tid]); // warp based random class

    if (migration == 0) // initialise population
    {
        initialise<tile_sz>(g, rnd, &island_population[island_idx * dim], dim, lower_bound, upper_bound);
        benchmarks_warp::evaluate<tile_sz>(g, function, &island_population[island_idx * dim], this_y, this_z, &island_fitness[island_idx], dim, shift, rotate, shuffle);
        if (g.thread_rank() == 0)
            check_solution(&island_fitness[island_idx], sol, island_idx);
    }

    thread_block.sync();

    for (int i = 1; i <= evaluations && !sol->solution_found; i++) // evaluation loop
    {
        strategies_warp::rand_one_binary<tile_sz>(g, rnd, island_population, this_offspring, dim, popsize, CR, F, lower_bound, upper_bound);
        benchmarks_warp::evaluate<tile_sz>(g, function, this_offspring, this_y, this_z, this_offspring_fitness, dim, shift, rotate, shuffle);
        thread_block.sync();
        replacement<tile_sz>(g, &island_population[island_idx * dim], &island_fitness[island_idx], this_offspring, this_offspring_fitness, dim, sol);
        thread_block.sync();
    }

    if (island_idx < num_migrants) // update migrants
    {
        int migrant_idx = blockIdx.x * num_migrants + island_idx;
        int good_idx = selection_warp::tournament_selection<tile_sz>(g, rnd, island_population, island_fitness, dim, popsize, 2, num_migrants, false);
        g.sync();
        for (int i = g.thread_rank(); i < dim; i += g.size())
            migrants[migrant_idx * dim + i] = island_population[good_idx * dim + i];
        if (g.thread_rank() == 0)
            migrant_fitness[migrant_idx] = island_fitness[good_idx];
    }
}

template <int tile_sz>
__global__ void migration_kernel(
    float *__restrict__ population,
    float *__restrict__ fitness,
    float *__restrict__ migrants,
    float *__restrict__ migrant_fitness,
    const int num_migrants,
    const int dim,
    const int popsize,
    const int islands,
    curandState *states)
{
    const auto thread_block = cg::this_thread_block();
    const auto g = cg::tiled_partition<tile_sz>(thread_block); // splits thread_block into thread block tiles of tile_sz
    const int block_tid = thread_block.thread_rank(); // overall thread number
    const int island_idx = g.meta_group_rank(); // population individual number
    if (island_idx >= popsize) // any extra threads return
        return;
    cu_random<tile_sz> rnd(&states[blockIdx.x * blockDim.x + block_tid]); // warp based random class

    int next_island = (blockIdx.x + 1) % islands; // next island in unidirectional ring 

    float *island_population = &population[blockIdx.x * popsize * dim];
    float *island_fitness = &fitness[blockIdx.x * popsize];

    if (island_idx < num_migrants)
    {
        int migrant_idx = next_island * num_migrants + island_idx;
        int replace_idx = selection_warp::tournament_selection<tile_sz>(g, rnd, island_population, island_fitness, dim, popsize, 2, num_migrants, true);
        g.sync();
        for (int i = g.thread_rank(); i < dim; i += g.size())
            island_population[replace_idx * dim + i] = migrants[migrant_idx * dim + i];
        if (g.thread_rank() == 0)
            island_fitness[replace_idx] = migrant_fitness[migrant_idx];
    }
}

int main(int argc, char **argv)
{
    if (argc != 8)
    {
        std::cout << "Usage: #OBJFUNC #DIMENSION #POPSIZE #ISLANDS #RUNS #EVAL_BET_MIG PLOT" << std::endl;
        exit(0);
    }
    int function_number = atoi(argv[1]);
    const int dim = atoi(argv[2]);
    int island_popsize = atoi(argv[3]);
    int threads = 32;
    // if (island_popsize == 32)
    //     threads = 16;
    // else
    //     threads = 32;
    int islands = atoi(argv[4]);
    int runs = atoi(argv[5]);
    int eval_between_migration = atoi(argv[6]);
    int plot_data = atoi(argv[7]);
    int evaluations;
    if (dim == 50)
        evaluations = 5000000;
    else if (dim == 100)
        evaluations = 10000000;
    const float lower_bound = -100.0f;
    const float upper_bound = 100.0f;

    int num_migrants = island_popsize / 10 + 1;
    const float F = 0.5;
    const float CR = 0.3;

    // set the CUDA device, in this case the first GPU
    int device = 0;
    cudaDeviceProp device_prop;
    cuda_error_check(cudaGetDeviceProperties(&device_prop, device));
    cuda_error_check(cudaSetDevice(device));

    cuBenchmarkData cuBD(function_number, dim);

    const unsigned long seed = 0;
    
    // create launch parameters for CUDA kernels
    dim3 grid_size(islands);
    dim3 block_size(threads * island_popsize);
    dim3 migrant_block_size(threads * num_migrants);

    // initialise CPU side memory
    std::vector<float> best_log;
    std::vector<float> time_log;
    std::vector<int> eval_log;

    // initialise data structures for CPU and CUDA data
    cuvector<float> population(island_popsize * islands * dim);
    cuvector<float> fitness(island_popsize * islands);

    solution h_solution, *d_solution;
    cuda_error_check(cudaMalloc((void **)&d_solution, sizeof(solution)));

    // initialise memory for CUDA data
    curandState *d_random_states;
    cuda_error_check(cudaMalloc((void **)&d_random_states, island_popsize * islands * threads * sizeof(curandState)));
    float *d_migrants, *d_migrant_fitness;
    cuda_error_check(cudaMalloc((void **)&d_migrants, num_migrants * islands * dim * sizeof(float)));
    cuda_error_check(cudaMalloc((void **)&d_migrant_fitness, num_migrants * islands * sizeof(float)));

    // initialise the device random states
    initialise_curand_kernel<<<islands, block_size>>>(d_random_states, seed);
    cuda_error_check(cudaPeekAtLastError());
    cuda_error_check(cudaDeviceSynchronize());

    for (int i = 0; i < runs; i++)
    {
        // reset solution
        cuda_error_check(cudaMemset(d_solution, 0, sizeof(solution)));

        std::ofstream("data/all_pop_data.txt"); // this will wipe the files
        std::ofstream("data/all_fitness_data.txt");
        // now open them in append mode
        auto population_file = std::ofstream("data/all_pop_data.txt", std::ofstream::out | std::ofstream::app); // now open in append mode
        auto fitness_file = std::ofstream("data/all_fitness_data.txt", std::ofstream::out | std::ofstream::app);

        std::vector<std::vector<float>> fitness_vector(islands);

        int evals = 0;
        int m = 0;
        size_t shared_bytes = (island_popsize + island_popsize * next_power_of_two(dim) * 3) * sizeof(float);

        // if we need more more shared memory than default, request more
        // std::cout << "Require: " << shared_bytes << " and default is " << device_prop.sharedMemPerBlock << std::endl; 
        // std::cout << device_prop.sharedMemPerBlockOptin << std::endl;
        if (shared_bytes >= device_prop.sharedMemPerBlock)
        {
            switch (threads)
            {
            case 32:
                cuda_error_check(cudaFuncSetAttribute(differential_evolution_kernel<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
                break;
            case 16:
                cuda_error_check(cudaFuncSetAttribute(differential_evolution_kernel<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
                break;
            case 8:
                cuda_error_check(cudaFuncSetAttribute(differential_evolution_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
                break;
            }
        }

        // setup cuda timer to time the algorithm
        float time_ms;
        cudaEvent_t start, stop;
        cuda_error_check(cudaEventCreate(&start));
        cuda_error_check(cudaEventCreate(&stop));
        cuda_error_check(cudaEventRecord(start, 0));

        do
        {
            switch (threads)
            {
            case 32:
                differential_evolution_kernel<32><<<grid_size, block_size, shared_bytes>>>(
                    function_number,
                    cuBD.get_rotate_transpose_ptr(),
                    cuBD.get_shift_ptr(),
                    cuBD.get_shuffle_ptr(),
                    population.get_device_ptr(),
                    fitness.get_device_ptr(),
                    d_migrants,
                    d_migrant_fitness,
                    F,
                    CR,
                    num_migrants,
                    dim,
                    island_popsize,
                    lower_bound,
                    upper_bound,
                    eval_between_migration,
                    m,
                    d_random_states,
                    d_solution);
                break;
            case 16:
                differential_evolution_kernel<16><<<grid_size, block_size, shared_bytes>>>(
                    function_number,
                    cuBD.get_rotate_transpose_ptr(),
                    cuBD.get_shift_ptr(),
                    cuBD.get_shuffle_ptr(),
                    population.get_device_ptr(),
                    fitness.get_device_ptr(),
                    d_migrants,
                    d_migrant_fitness,
                    F,
                    CR,
                    num_migrants,
                    dim,
                    island_popsize,
                    lower_bound,
                    upper_bound,
                    eval_between_migration,
                    m,
                    d_random_states,
                    d_solution);
                break;
            case 8:
                differential_evolution_kernel<8><<<grid_size, block_size, shared_bytes>>>(
                    function_number,
                    cuBD.get_rotate_transpose_ptr(),
                    cuBD.get_shift_ptr(),
                    cuBD.get_shuffle_ptr(),
                    population.get_device_ptr(),
                    fitness.get_device_ptr(),
                    d_migrants,
                    d_migrant_fitness,
                    F,
                    CR,
                    num_migrants,
                    dim,
                    island_popsize,
                    lower_bound,
                    upper_bound,
                    eval_between_migration,
                    m,
                    d_random_states,
                    d_solution);
                break;
            }

            cuda_error_check(cudaPeekAtLastError());
            cuda_error_check(cudaDeviceSynchronize());

            if (plot_data)
            {
                // copy population and statistical data from gpu to cpu
                fitness.cpu();
                population.cpu();

                for (int j = 0; j < islands; j++)
                {
                    fitness_file << function_number << " " << j << " " << island_popsize << " ";
                    for (int k = 0; k < island_popsize; k++)
                        fitness_file << fitness[j * island_popsize + k] << " ";
                    fitness_file << std::endl;

                    population_file << function_number << " " << j << " " << island_popsize << " " << dim << " ";
                    for (int k = 0; k < island_popsize; k++)
                        for (int p = 0; p < dim; p++)
                            population_file << population[j * island_popsize * dim + k * dim + p] << " ";
                    population_file << std::endl;
                }
            }
            // grab solution struct and see whether we stop
            cuda_error_check(cudaMemcpy(&h_solution, d_solution, sizeof(solution), cudaMemcpyDeviceToHost));
            if (h_solution.solution_found > 0)
            {
                evals += h_solution.solution_iteration * island_popsize * islands;
                break;
            }
            evals += eval_between_migration * island_popsize * islands;
            m++;
            if (islands > 1 && evals < evaluations)
            {
                switch (threads)
                {
                case 32:
                    migration_kernel<32><<<grid_size, block_size>>>(
                        population.get_device_ptr(),
                        fitness.get_device_ptr(),
                        d_migrants,
                        d_migrant_fitness,
                        num_migrants,
                        dim,
                        island_popsize,
                        islands,
                        d_random_states);
                    break;
                case 16:
                    migration_kernel<16><<<grid_size, block_size>>>(
                        population.get_device_ptr(),
                        fitness.get_device_ptr(),
                        d_migrants,
                        d_migrant_fitness,
                        num_migrants,
                        dim,
                        island_popsize,
                        islands,
                        d_random_states);
                    break;
                case 8:
                    migration_kernel<8><<<grid_size, block_size>>>(
                        population.get_device_ptr(),
                        fitness.get_device_ptr(),
                        d_migrants,
                        d_migrant_fitness,
                        num_migrants,
                        dim,
                        island_popsize,
                        islands,
                        d_random_states);
                    break;
                }
                cuda_error_check(cudaPeekAtLastError());
                cuda_error_check(cudaDeviceSynchronize());
            }
        } while (evals < evaluations);

        // stop the timer
        cuda_error_check(cudaEventRecord(stop, 0));
        cuda_error_check(cudaEventSynchronize(stop));
        cuda_error_check(cudaEventElapsedTime(&time_ms, start, stop));
        auto time_sec = time_ms / 1000;

        // get population data
        population.cpu();
        fitness.cpu();

        auto best = *std::min_element(fitness.begin(), fitness.end());
        std::cout << "Iteration: " << i << " Function: " << function_number << " Best fitness: " << best << " Time: " << time_sec << std::endl;
        best_log.push_back(best);
        time_log.push_back(time_sec);
        eval_log.push_back(evals);
    }
    auto min_fitness = *std::min_element(best_log.begin(), best_log.end());
    auto max_fitness = *std::max_element(best_log.begin(), best_log.end());

    log_data(function_number, dim, island_popsize, islands, "degi_16", best_log, time_log, eval_log);

    float mean, stdev;
    mean_and_stdev(best_log, mean, stdev);
    float time_mean, time_stdev;
    mean_and_stdev(time_log, time_mean, time_stdev);
    float gen_mean, gen_stdev;
    mean_and_stdev(eval_log, gen_mean, gen_stdev);

    std::cout << function_number << ", " << min_fitness << ", " << max_fitness << ", " << mean
              << ", " << stdev << ", " << time_mean << ", " << time_stdev << ", "
              << gen_mean << ", " << gen_stdev << std::endl;
    cudaFree(d_random_states);
    cudaFree(d_migrants);
    cudaFree(d_migrant_fitness);
}