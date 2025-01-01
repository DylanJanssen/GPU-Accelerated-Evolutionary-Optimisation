#include <cooperative_groups.h> 
#include <iostream> 
#include <algorithm> 
#include <iterator>

#include "cuda/warp/cu_random.cuh"
#include "../numerical_benchmark/cuda/cuda_benchmarks_warp.cuh"
#include "cuda/warp/strategies.cuh"
#include "../common/cuvector.cuh"
#include "../common/cuda_err.cuh"
#include "../common/helper_functions.hpp"
#include "cuda/solution.cuh"
#include "cuda/warp/selection.cuh"

#define NUM_STRATEGIES 4

namespace cg = cooperative_groups; 

__host__ __device__ __forceinline__ 
int log2_ceil(int value) 
{
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}

__host__ __device__ __forceinline__ 
int next_power_of_two(int x)
{
    int log2_elements = log2_ceil(x);
    return 1 << log2_elements;
}


/***************************
 * DIFFERENTIAL EVOLUTION  *
 ****************************/
template <int tile_sz>
__global__ void 
__launch_bounds__(1024,1)
differential_evolution_kernel(
    int function_number, 
    float *__restrict__ rotate,
    float *__restrict__ shift,
    int *__restrict__ shuffle,
    float *__restrict__ population,
    float *__restrict__ fitness,
    float *__restrict__ offspring,
    float *__restrict__ offspring_fitness,
    float *__restrict__ migrants,
    float *__restrict__ migrant_fitness,
    float *__restrict__ scratch_space,
    const int num_migrants,
    const int dim,
    const int popsize,
    const float lower_bound,
    const float upper_bound,
    const int evaluations,
    const int migration,
    const int islands,
    float *probability_data,
    float *CRm_data,
    curandState *__restrict__ states,
    const int LR,
    solution *sol)
{
    __shared__ float prob[NUM_STRATEGIES];
    __shared__ int best, ns[NUM_STRATEGIES], nf[NUM_STRATEGIES];
    __shared__ float CRm[NUM_STRATEGIES];

    // cooperative group for all the threads of the thread block 
    cg::thread_block block = cg::this_thread_block(); 
    // cooperative group for each individual 
    cg::thread_block_tile<tile_sz> g = cg::tiled_partition<tile_sz>(block);
    // cooperative groups of 32 threads (warp size) for other operations 
    cg::thread_block_tile<32> pop_g = cg::tiled_partition<32>(block); 

    cu_random<tile_sz> rnd(&states[block.group_index().x * block.num_threads() + block.thread_rank()]);    
    int block_tid = block.thread_rank();
    int island_idx = g.meta_group_rank();

    if (island_idx >= popsize) // any extra threads return
        return;

    // float *island_scratch_space;
    // island_scratch_space = &scratch_space[2 * blockIdx.x * popsize * dim];
    
    float *island_population = &population[blockIdx.x * popsize * dim];
    float *island_fitness = &fitness[blockIdx.x * popsize];

    // shared memory 
    extern __shared__ char smem[];
    int memory_size = next_power_of_two(dim); 
    float *y = (float *)smem; // dynamically allocated shared memory 
    float *this_y = &y[island_idx * memory_size]; // used for objective functions 
    float *z = &y[popsize * memory_size];
    float *this_z = &z[island_idx * memory_size]; // used for objective functions 
    float *offspring2 = &z[popsize * memory_size];
    float *this_offspring = &offspring2[island_idx * memory_size];
    float *offspring_fitness2 = &offspring2[popsize * memory_size];
    float *this_offspring_fitness = &offspring_fitness2[island_idx];
    float *CR_log = &offspring_fitness2[popsize];

    // float *this_y = &island_scratch_space[island_idx * dim * 2];
    // float *this_z = &this_y[dim];

    
    // float *offspring_island_population = &offspring[blockIdx.x * popsize * dim];
    // float *this_offspring = offspring_island_population + island_idx * dim; 
    // float *offspring_island_fitness = &offspring_fitness[blockIdx.x * popsize];
    // float *this_offspring_fitness = offspring_island_fitness + island_idx; 
    
    // set up this islands mutation probabilities, ns/nf arrays, and CRm values.
    if (threadIdx.x < NUM_STRATEGIES)
    {
        ns[threadIdx.x] = 0; 
        nf[threadIdx.x] = 0; 
        if (migration == 0) 
        {
            prob[threadIdx.x] = 1.0f / NUM_STRATEGIES; 
            CRm[threadIdx.x] = 0.5f; 
        }
        else 
        {
            prob[threadIdx.x] = probability_data[blockIdx.x * evaluations * NUM_STRATEGIES + (evaluations-1) * NUM_STRATEGIES + threadIdx.x];
            CRm[threadIdx.x] = CRm_data[blockIdx.x * evaluations * NUM_STRATEGIES + (evaluations-1) * NUM_STRATEGIES + threadIdx.x];
        }
    }
    // could algo do g.thread_rank() < NUM_STRATEGIES however if thread group is say 2 with 4 strategies...
    if (g.thread_rank() == 0) // this could be optimised using block tid, that way just a few warps
        for (int i = 0; i < NUM_STRATEGIES; i++)
            CR_log[island_idx * NUM_STRATEGIES + i] = 0.0f; // initialise this individuals log to 0

    if (migration == 0) // initialise population
    {
        for (int i = g.thread_rank(); i < dim; i += g.num_threads())
            island_population[island_idx * dim + i] = rnd.uniform(lower_bound, upper_bound);
        g.sync();
        benchmarks_warp::evaluate<tile_sz>(g, function_number, &island_population[island_idx * dim], this_y, this_z, &island_fitness[island_idx], dim, shift, rotate, shuffle);

        if (g.thread_rank() == 0)
            if (island_fitness[island_idx] < 10e-8f)
            {
                island_fitness[island_idx] = 0.0f;
                int val = atomicAdd(&sol->solution_found, 1); 
                if (val == 0) 
                {
                    sol->solution_idx = island_idx + blockIdx.x * popsize; 
                    sol->solution_iteration = 1; 
                }                
                return; 
            }
    }

    block.sync();

    float choice, F, CR;
    int k;

    for (int i = 2; i <= evaluations; i++)
    {
        choice = rnd.uniform(g);
        k = NUM_STRATEGIES-1;
        float sum = prob[0]; 

        for (int j = 1; j < NUM_STRATEGIES; j++)
        {
            if (choice < sum)
            {
                k = j-1; 
                break; 
            }
            sum += prob[j]; 
        }

        F = rnd.normal(g, 0.5f, 0.3f);
        do {
            CR = rnd.normal(g, CRm[k], 0.1f);
        } while (CR < 0.0f || CR > 1.0f);

        // warp level reduction to find best parent (only works for up to 32 individuals per island - actually since we use half the threads, would work up to 64)
        if (pop_g.meta_group_rank() == 0)
        {
            int local_best = pop_g.thread_rank(); 
            int tmp_ind; 
            for (int offset = pop_g.size() / 2; offset > 0; offset /= 2) // parallel reduction
            {
                tmp_ind = pop_g.shfl_down(local_best, offset);
                if (pop_g.thread_rank() + offset < popsize && island_fitness[tmp_ind] < island_fitness[local_best])
                    local_best = tmp_ind; 
            }
            if (pop_g.thread_rank() == 0)
                best = local_best;
        }
        
        block.sync();
        
        if (k == 0)
            strategies_warp::rand_one_binary<tile_sz>(g, rnd, island_population, this_offspring, dim, popsize, CR, F, lower_bound, upper_bound);
        else if (k == 1)
            strategies_warp::rand_two_binary<tile_sz>(g, rnd, island_population, this_offspring, dim, popsize, CR, F, lower_bound, upper_bound);
        else if (k == 2)
            strategies_warp::rand_to_best_two_binary<tile_sz>(g, rnd, island_population, this_offspring, best, dim, popsize, CR, F, lower_bound, upper_bound);
        else
            strategies_warp::current_to_rand_one<tile_sz>(g, rnd, island_population, this_offspring, best, dim, popsize, CR, F, lower_bound, upper_bound);

        g.sync();

        benchmarks_warp::evaluate<tile_sz>(g, function_number, this_offspring, this_y, this_z, this_offspring_fitness, dim, shift, rotate, shuffle);
        
        block.sync();

        if (sol->solution_found > 0) // early return if solution has been found 
            return; 
        
        if (*this_offspring_fitness < island_fitness[island_idx])
        {
            if (g.thread_rank() == 0) // update CR log, ns and fitness
            {
                CR_log[island_idx * NUM_STRATEGIES + k] += CR;
                atomicAdd(&ns[k], 1);
                island_fitness[island_idx] = *this_offspring_fitness;
                check_solution(&island_fitness[island_idx], sol, g.meta_group_rank());
            }
            for (int m = g.thread_rank(); m < dim; m += g.num_threads())
                island_population[island_idx * dim + m] = this_offspring[m];
        }
        else
        {
            if (g.thread_rank() == 0)
                atomicAdd(&nf[k], 1);
        }

        block.sync();

        if (i % LR == 0) // mean update
        {
            // parallel reduction
            for (int j = popsize / 2; j > 0; j /= 2)
            {
                if (block_tid < j)
                {
                    for (int k = 0; k < NUM_STRATEGIES; k++)
                    {
                        CR_log[block_tid * NUM_STRATEGIES + k] += CR_log[(block_tid + j) * NUM_STRATEGIES + k];
                        CR_log[(block_tid + j) * NUM_STRATEGIES + k] = 0.0f;
                    }
                }
                block.sync();
            }
            if (threadIdx.x < NUM_STRATEGIES)
            {
                // update mean 
                if (CR_log[threadIdx.x] + ns[threadIdx.x] > 0)
                    CRm[threadIdx.x] = CR_log[threadIdx.x] / ns[threadIdx.x];
                else
                    CRm[threadIdx.x] = 0.01f;
                CR_log[threadIdx.x] = 0.0f;
                // update prob and reset ns/nf 
                if (ns[threadIdx.x] + nf[threadIdx.x] > 0)
                    prob[threadIdx.x] = (float)ns[threadIdx.x] / (float)(ns[threadIdx.x] + nf[threadIdx.x]) + 0.01f;
                else
                    prob[threadIdx.x] = 0.01f;
                ns[threadIdx.x] = 0; 
                nf[threadIdx.x] = 0; 
            }
            block.sync(); 
            if (block_tid == 0)
            {
                float sum = 0.0f;
                for (int j = 0; j < NUM_STRATEGIES; j++)
                    sum += prob[j];
                for (int j = 0; j < NUM_STRATEGIES; j++)
                    prob[j] /= sum;
            }
        }
        block.sync();

        if (threadIdx.x < NUM_STRATEGIES) 
        {
            probability_data[blockIdx.x * evaluations * NUM_STRATEGIES + (i - 1) * NUM_STRATEGIES + threadIdx.x] = prob[threadIdx.x];
            CRm_data[blockIdx.x * evaluations * NUM_STRATEGIES + (i - 1) * NUM_STRATEGIES + threadIdx.x] = CRm[threadIdx.x];
        }
        block.sync();

    } // END of evaluations loop

    if (island_idx < num_migrants)
    {
        int migrant_idx = blockIdx.x * num_migrants + island_idx;
        int good_idx = selection_warp::tournament_selection<tile_sz>(
            g,
            rnd,
            island_population,
            island_fitness,
            dim,
            popsize,
            8,
            num_migrants,
            false);

        g.sync();
        
        for (int i = g.thread_rank(); i < dim; i += g.num_threads())
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
    cu_random<tile_sz> rnd(&states[blockIdx.x * blockDim.x + cg::this_thread_block().thread_rank()]);
    auto g = cg::tiled_partition<tile_sz>(cg::this_thread_block());
    int block_tid = cg::this_thread_block().thread_rank();
    int island_idx = block_tid / g.num_threads();
    int next_island = (blockIdx.x + 1) % islands;

    float *island_population = &population[blockIdx.x * popsize * dim];
    float *island_fitness = &fitness[blockIdx.x * popsize];

    if (island_idx < num_migrants)
    {
        int migrant_idx = next_island * num_migrants + island_idx;
        int replace_idx = selection_warp::tournament_selection<tile_sz>(
            g,
            rnd,
            island_population,
            island_fitness,
            dim,
            popsize,
            8,
            num_migrants,
            true);
        g.sync();

        for (int i = g.thread_rank(); i < dim; i += g.num_threads())
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
    const std::string algorithm_name("degias_16_shared");
    int function_number = atoi(argv[1]);
    const int dim = atoi(argv[2]);
    int island_popsize = atoi(argv[3]);
    int islands = atoi(argv[4]);
    int runs = atoi(argv[5]);
    int eval_between_migration = atoi(argv[6]);
    int plot_data = atoi(argv[7]);
    int LR = 50;
    int evaluations; 

    std::stringstream ss;
    std::string func = (function_number > 9 ? "" : "0") + std::to_string(function_number);
    ss << "output_data/" << algorithm_name << "/F" << func << "_D" << dim << "_P" << island_popsize << "_I" << islands;
    

    if (dim == 50) 
        evaluations = 5000000; 
    else if (dim == 100) 
        evaluations = 10000000;

    int threads = 16; 
    // if (island_popsize >= 25)
    //     threads = 16; 

    const float lower_bound = -100.0f;
    const float upper_bound = 100.0f;
    int num_migrants = island_popsize / 10 + 1;
    
    // set the CUDA device, in this case the first GPU
    int device = 0;
    cudaDeviceProp device_prop;
    cuda_error_check(cudaGetDeviceProperties(&device_prop, device));
    cuda_error_check(cudaSetDevice(device));

    cuBenchmarkData cuBD(function_number, dim); 
    
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
    cuvector<float> probability_data(eval_between_migration * islands * NUM_STRATEGIES); 
    cuvector<float> CRm_data(eval_between_migration * islands * NUM_STRATEGIES); 

    solution h_solution, *d_solution; 
    cuda_error_check(cudaMalloc((void**)&d_solution, sizeof(solution)));
    

    // initialise memory for CUDA data 
    curandState *d_random_states;
    cuda_error_check(cudaMalloc((void **)&d_random_states, island_popsize * islands * threads * sizeof(curandState)));
    float *d_offspring, *d_offspring_fitness, *d_migrants, *d_migrant_fitness, *d_scratch_space;
    cuda_error_check(cudaMalloc((void **)&d_scratch_space, 2 * island_popsize * islands * dim * sizeof(float)));
    cuda_error_check(cudaMalloc((void **)&d_offspring, island_popsize * islands * dim * sizeof(float)));
    cuda_error_check(cudaMalloc((void **)&d_offspring_fitness, island_popsize * islands * sizeof(float)));
    cuda_error_check(cudaMalloc((void **)&d_migrants, num_migrants * islands * dim * sizeof(float)));
    cuda_error_check(cudaMalloc((void **)&d_migrant_fitness, num_migrants * islands * sizeof(float)));

    // initialise the device random states
    const unsigned long seed = 0;
    initialise_curand_kernel<<<islands, block_size>>>(d_random_states, seed);
    cuda_error_check(cudaPeekAtLastError());
    cuda_error_check(cudaDeviceSynchronize());

    for (int i = 0; i < runs; i++)
    {
        // reset solution 
        cuda_error_check(cudaMemset(d_solution, 0, sizeof(solution))); 

        
        std::ofstream output(ss.str() + "_probability_data.txt");

        std::ofstream(ss.str() + "_all_pop_data.txt"); // this will wipe the files
        std::ofstream(ss.str() + "_all_fitness_data.txt");
        // now open them in append mode
        auto population_file = std::ofstream(ss.str() + "_all_pop_data.txt", std::ofstream::out | std::ofstream::app); // now open in append mode
        auto fitness_file = std::ofstream(ss.str() + "_all_fitness_data.txt", std::ofstream::out | std::ofstream::app);

        std::vector<std::vector<float>> fitness_vector(islands);
        std::vector<std::vector<float>> CRm_vector(islands);
        std::vector<std::vector<float>> probability_vector(islands);

        int evals = 0;
        int m = 0;
        size_t shared_bytes = 
            (island_popsize + // offspring fitness
             island_popsize * next_power_of_two(dim) * 3 + // y, z, offspring arrays
             island_popsize * NUM_STRATEGIES) // CRm array
             * sizeof(float);
        // std::cout << "Shared memory required: " << shared_bytes << std::endl; 
        // if we need more more shared memory than default, request more 
        if (shared_bytes >= device_prop.sharedMemPerBlock)
        {
            switch(threads) 
            {
            case 32:
                cudaFuncSetAttribute(differential_evolution_kernel<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
                break;
            case 16:
                cudaFuncSetAttribute(differential_evolution_kernel<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
                break;
            case 8:
                cudaFuncSetAttribute(differential_evolution_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
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
            switch(threads) 
            {
            case 32: 
                differential_evolution_kernel<32><<<grid_size, block_size, shared_bytes>>>(
                    function_number, 
                    cuBD.get_rotate_transpose_ptr(), 
                    cuBD.get_shift_ptr(),
                    cuBD.get_shuffle_ptr(),
                    population.get_device_ptr(),
                    fitness.get_device_ptr(),
                    d_offspring,
                    d_offspring_fitness,
                    d_migrants,
                    d_migrant_fitness,
                    d_scratch_space,
                    num_migrants,
                    dim,
                    island_popsize,
                    lower_bound,
                    upper_bound,
                    eval_between_migration,
                    m,
                    islands,
                    probability_data.get_device_ptr(),
                    CRm_data.get_device_ptr(),
                    d_random_states,
                    LR,
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
                    d_offspring,
                    d_offspring_fitness,
                    d_migrants,
                    d_migrant_fitness,
                    d_scratch_space,
                    num_migrants,
                    dim,
                    island_popsize,
                    lower_bound,
                    upper_bound,
                    eval_between_migration,
                    m,
                    islands,
                    probability_data.get_device_ptr(),
                    CRm_data.get_device_ptr(),
                    d_random_states,
                    LR,
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
                    d_offspring,
                    d_offspring_fitness,
                    d_migrants,
                    d_migrant_fitness,
                    d_scratch_space,
                    num_migrants,
                    dim,
                    island_popsize,
                    lower_bound,
                    upper_bound,
                    eval_between_migration,
                    m,
                    islands,
                    probability_data.get_device_ptr(),
                    CRm_data.get_device_ptr(),
                    d_random_states,
                    LR,
                    d_solution);
                    break; 
            }
            cuda_error_check(cudaPeekAtLastError());
            cuda_error_check(cudaDeviceSynchronize());

            if (plot_data)
            {
                // copy population and statistical data from gpu to cpu 
                probability_data.cpu(); 
                CRm_data.cpu(); 
                fitness.cpu(); 
                population.cpu(); 

                for (int j = 0; j < islands; j++)
                {
                    std::copy(&probability_data[j * eval_between_migration * NUM_STRATEGIES], &probability_data[(j + 1) * eval_between_migration * NUM_STRATEGIES], back_inserter(probability_vector[j]));
                    std::copy(&CRm_data[j * eval_between_migration * NUM_STRATEGIES], &CRm_data[(j + 1) * eval_between_migration * NUM_STRATEGIES], back_inserter(CRm_vector[j]));

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
            // std::cout << evals << std::endl; 
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

        if (plot_data)
        {
            std::stringstream ss;
            std::string func = (function_number > 9 ? "" : "0") + std::to_string(function_number);
            ss << "output_data/" << algorithm_name << "/F" << func << "_D" << dim << "_P" << island_popsize << "_I" << islands;
            std::ofstream output(ss.str() + "_probability_data.txt");
            
            for (auto island : probability_vector)
            {
                std::copy(island.begin(), island.end(), std::ostream_iterator<float>(output, ","));
                output << "\n";
            }
            output.close();
            output.clear();
            output.open(ss.str() + "_CRm_data.txt");
            for (auto island : CRm_vector)
            {
                std::copy(island.begin(), island.end(), std::ostream_iterator<float>(output, ","));
                output << "\n";
            }
            output.close();
        }
        // get population data
        population.cpu(); 
        fitness.cpu(); 
        
        auto best = *std::min_element(fitness.begin(), fitness.end());
        std::cout << "Function: " << function_number << " Best fitness: " << best << " Time: " << time_sec << std::endl;
        best_log.push_back(best);
        time_log.push_back(time_sec);
        eval_log.push_back(evals);
    }
    auto min_fitness = *std::min_element(best_log.begin(), best_log.end());
    auto max_fitness = *std::max_element(best_log.begin(), best_log.end());
    
    log_data(function_number, dim, island_popsize, islands, algorithm_name, best_log, time_log, eval_log); 

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