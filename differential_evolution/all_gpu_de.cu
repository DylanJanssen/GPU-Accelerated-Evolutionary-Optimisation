#include <cooperative_groups.h>

#include <iostream>
#include <algorithm>
#include <numeric>

#include "../common/helper_functions.hpp"
#include "../common/cuda_err.cuh"
#include "../common/cuvector.cuh"
#include "../numerical_benchmark/cuda/cuda_benchmarks.cuh"
#include "cuda/solution.cuh"
#include "cuda/block/random.cuh"

namespace cg = cooperative_groups;

__global__ void population_initialisation(
    float *__restrict__ population,
    const int dim,
    const int popsize,
    const float lower_bound,
    const float upper_bound,
    curandState *__restrict__ states)
{
    if (blockIdx.x >= popsize)
        return;
    float *this_x = population + blockIdx.x * dim;
    curandState *state = &states[blockIdx.x * blockDim.x + threadIdx.x];

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        this_x[i] = lower_bound + curand_uniform(state) * (upper_bound - lower_bound);
}

__global__ void select_random_trial_vectors(
    int *__restrict__ indices,
    const int popsize,
    curandState *__restrict__ states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= popsize)
        return;
    curandState *state = &states[blockIdx.x * blockDim.x + threadIdx.x];
    int a = mutually_exclusive_random_int(0, popsize, state);
    int b = mutually_exclusive_random_int(0, popsize, state, a);
    int c = mutually_exclusive_random_int(0, popsize, state, a, b);
    indices[idx] = a;
    indices[idx + popsize] = b;
    indices[idx + popsize + popsize] = c;
}

__global__ void trial_vector_generation(
    float *__restrict__ population,
    float *__restrict__ offspring,
    const int dim,
    const int popsize,
    int *__restrict__ indices,
    const float crossover_rate,
    const float f,
    const float lower_bound,
    const float upper_bound,
    curandState *__restrict__ states)
{
    int idx = blockIdx.x;
    if (idx >= popsize)
        return;
    float *this_offspring = &offspring[idx * dim];
    int a = indices[idx];
    int b = indices[idx + popsize];
    int c = indices[idx + popsize + popsize];
    curandState *state = &states[blockIdx.x * blockDim.x + threadIdx.x];
    __shared__ int j;
    if (threadIdx.x == 0)
        j = mutually_exclusive_random_int(0, dim, state);
    __syncthreads();
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        if (curand_uniform(state) < crossover_rate || i == j)
        {
            this_offspring[i] = population[a * dim + i] + f * (population[b * dim + i] - population[c * dim + i]);
            if (this_offspring[i] < lower_bound || this_offspring[i] > upper_bound)
                this_offspring[i] = lower_bound + curand_uniform(state) * (upper_bound - lower_bound);
        }
        else
            this_offspring[i] = population[idx * dim + i];
}

__global__ void replacement(
    float *__restrict__ population,
    float *__restrict__ fitness,
    float *__restrict__ offspring,
    float *__restrict__ offspring_fitness,
    const int dim,
    const int popsize,
    solution *sol)
{
    int idx = blockIdx.x;
    if (idx >= popsize)
        return;
    float *this_x = &population[idx * dim];
    float *this_offspring = &offspring[idx * dim];
    if (offspring_fitness[idx] < fitness[idx])
    {
        for (int i = threadIdx.x; i < dim; i += blockDim.x)
            this_x[i] = this_offspring[i];
        if (threadIdx.x == 0)
        {
            fitness[idx] = offspring_fitness[idx];
            check_solution(&fitness[idx], sol, idx);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        std::cout << "Usage: ./program_name #OBJFUNC #DIMENSION #POPSIZE #RUNS" << std::endl;
        exit(0);
    }
    const int function_number = stoi(argv[1]);
    const int dim = stoi(argv[2]);
    const int popsize = stoi(argv[3]);
    const int total_runs = stoi(argv[4]);

    int max_evaluations = 10000000; 
    if (dim == 50)
        max_evaluations = 5000000;
    
    std::cout << "Running all-gpu de with popsize=" << popsize << std::endl; 
    
    const float lower_bound = -100.0f;
    const float upper_bound = 100.0f;
    const float F = 0.5;
    const float CR = 0.3;

    cuvector<float> x(popsize * dim);
    cuvector<float> fitness(popsize);
    cuvector<float> offspring(popsize * dim);
    cuvector<float> offspring_fitness(popsize);
    curandState *d_random_states;
    int threads = next_power_of_two(dim);
    int blocks = popsize;
    cuda_error_check(cudaMalloc((void **)&d_random_states, blocks * threads * sizeof(curandState)));
    const unsigned long seed = 0;
    initialise_curand_kernel<<<blocks, threads>>>(d_random_states, seed);
    cuda_error_check(cudaPeekAtLastError());
    cuda_error_check(cudaDeviceSynchronize());

    cudaStream_t stream[2];
    for (int i = 0; i < 2; i++)
        cudaStreamCreate(&stream[i]);

    // memory for random indices
    int *indices;
    int req_indices = 3;
    int n_threads = 128;
    int n_blocks = int_divide_up(popsize, n_threads);
    cuda_error_check(cudaMalloc((void **)&indices, popsize * req_indices * sizeof(int)));

    solution h_solution, *d_solution;
    cuda_error_check(cudaMalloc((void **)&d_solution, sizeof(solution)));

    // load benchmark function data 
    cuBenchmarkData cuBD(function_number, dim);

    std::vector<float> best_log;
    std::vector<float> time_log;
    std::vector<int> eval_log;

    // setup cuda timer to time the algorithm
    float time_ms;
    cudaEvent_t start, stop;
    cuda_error_check(cudaEventCreate(&start));
    cuda_error_check(cudaEventCreate(&stop));
    
    for (int runs = 1; runs <= total_runs; runs++)
    {
        // reset solution
        cuda_error_check(cudaMemset(d_solution, 0, sizeof(solution)));

        // start timer 
        cuda_error_check(cudaEventRecord(start, 0));

        population_initialisation<<<blocks, threads>>>(x.get_device_ptr(), dim, popsize, lower_bound, upper_bound, d_random_states);
        cuda_error_check(cudaPeekAtLastError());
        cuda_error_check(cudaDeviceSynchronize());

        evaluate(function_number, x.get_device_ptr(), fitness.get_device_ptr(), popsize, dim, cuBD.get_shift_ptr(), cuBD.get_rotate_transpose_ptr(), cuBD.get_shuffle_ptr());

        int evals = popsize;

        // begin evolutionary loop 
        while (evals < max_evaluations && h_solution.solution_found == 0)
        {
            select_random_trial_vectors<<<n_blocks, n_threads>>>(indices, popsize, d_random_states);

            cuda_error_check(cudaPeekAtLastError());
            cuda_error_check(cudaDeviceSynchronize());

            trial_vector_generation<<<blocks, threads, 0, stream[0]>>>(x.get_device_ptr(), offspring.get_device_ptr(), dim, popsize,
                                                                       indices, CR, F, lower_bound, upper_bound, d_random_states);
            // grab solution struct asynchronously and see whether we stop
            cuda_error_check(cudaMemcpyAsync(&h_solution, d_solution, sizeof(solution), cudaMemcpyDeviceToHost, stream[1]));

            cuda_error_check(cudaPeekAtLastError());
            cuda_error_check(cudaDeviceSynchronize());

            evaluate(function_number, offspring.get_device_ptr(), offspring_fitness.get_device_ptr(), popsize, dim, cuBD.get_shift_ptr(), cuBD.get_rotate_transpose_ptr(), cuBD.get_shuffle_ptr());

            replacement<<<blocks, threads>>>(
                x.get_device_ptr(), fitness.get_device_ptr(),
                offspring.get_device_ptr(), offspring_fitness.get_device_ptr(),
                dim, popsize, d_solution);

            cuda_error_check(cudaPeekAtLastError());
            cuda_error_check(cudaDeviceSynchronize());

            evals += popsize;
        }

        // stop the timer
        cuda_error_check(cudaEventRecord(stop, 0));
        cuda_error_check(cudaEventSynchronize(stop));
        cuda_error_check(cudaEventElapsedTime(&time_ms, start, stop));
        auto time_sec = time_ms / 1000; 

        fitness.cpu();
        
        auto best = *std::min_element(fitness.begin(), fitness.end());

        best_log.push_back(best);

        std::cout << "Function: " << function_number << " Best fitness: " << best << " Time: " << time_sec << std::endl;
        time_log.push_back(time_sec);
        eval_log.push_back(evals);
    }
    auto min_fitness = *std::min_element(best_log.begin(), best_log.end());
    auto max_fitness = *std::max_element(best_log.begin(), best_log.end());
    
    log_data(function_number, dim, popsize, "all_gpu_de", best_log, time_log, eval_log); 

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
    cudaFree(indices);
    for (int i = 0; i < 2; i++)
        cudaStreamDestroy(stream[i]);
}
