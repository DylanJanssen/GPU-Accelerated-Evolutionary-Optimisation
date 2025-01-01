// cuda includes
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cudaProfiler.h>
#include <curand.h>
#include <curand_kernel.h>

// c includes 
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <float.h>

#include "../common/cuda_err.cuh"

#ifdef __CUDACC__
#define L(x,y) __launch_bounds__(x,y)
#else
#define L(x,y)
#endif

#define FULL_MASK 0xffffffff // used as the active mask for warp primitives (32 bits to indicate to use all threads of warp)

typedef struct {
	int chromosome_length;
	int generations;
	int migrations;
	int total_generations;
	int number_of_islands;
	int island_population_size;
	int thread_group_size;
	int tournament_size;
	int total_population_size;
	int number_of_migrants;
	float min;
	float max;
	float mutation_probability;
	float crossover_probability;
	float replacement_probability;
	float solution;
	bool elitism;
	bool stop_sol_found;
	bool combinatorial;
	bool maximisation;
	bool log_data;
} parameters;


struct solution {
	int solution_found;
	int solution_generation;
	int solution_index;
	float best_fitness;
	long long int ms;
};


// typedef function pointer
typedef void (*objective_function)(void*, float*, unsigned, const parameters);

// fitness functions forward declarations 
__device__ void calculate_n_queens_fitness(void* __restrict__ x1, float* __restrict__ fitness, unsigned active_threads_mask, const parameters p);
__device__ void calculate_rosenbrock_fitness(void* __restrict__ x1, float* __restrict__ fitness, unsigned active_threads_mask, const parameters p);
__device__ void calculate_griewank_fitness(void* __restrict__ x1, float* __restrict__ fitness, unsigned active_threads_mask, const parameters p);
__device__ void calculate_weierstrass_fitness(void* __restrict__ x1, float* __restrict__ fitness, unsigned active_threads_mask, const parameters p);
__device__ void calculate_schwefel_fitness(void* __restrict__ x1, float* __restrict__ fitness, unsigned active_threads_mask, const parameters p);
__device__ void calculate_rastrigin_fitness(void* __restrict__ x1, float* __restrict__ fitness, unsigned active_threads_mask, const parameters p);

// declare global function pointers for device side functions 
__device__ objective_function d_calculate_n_queens_fitness = &calculate_n_queens_fitness;
__device__ objective_function d_calculate_rosenbrock_fitness = &calculate_rosenbrock_fitness;
__device__ objective_function d_calculate_griewank_fitness = &calculate_griewank_fitness;
__device__ objective_function d_calculate_weierstrass_fitness = &calculate_weierstrass_fitness;
__device__ objective_function d_calculate_schwefel_fitness = &calculate_schwefel_fitness;
__device__ objective_function d_calculate_rastrigin_fitness = &calculate_rastrigin_fitness;

// CURAND initialisation
__global__ void initialise_curand_kernel(
	curandState* __restrict__ state,
	const unsigned long seed)
{
	const unsigned int tid = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	curand_init(seed, tid, 0, &state[tid]);
}


// device function that generates a chromosome of random genes in the range [min, max]
template<typename T>
__device__ void generate_chromosome(
	T* __restrict__ individual,
	curandState* __restrict__ state,
	const parameters p)
{
	for (int i = threadIdx.x; i < p.chromosome_length; i += p.thread_group_size)
		individual[i] = p.min + curand_uniform(state) * (p.max - p.min);
}

// device function to determine the fitness of the given chromosome (p.chromosome_length-Queens)
__device__ void calculate_n_queens_fitness(
	void* __restrict__ x1,
	float* __restrict__ fitness,
	const unsigned active_threads_mask,
	const parameters p)
{
	int* x = (int*)x1;
	int clashes = 0;

	// calculate p.thread_group_size chunks of the function
	for (int i = threadIdx.x; i < p.chromosome_length; i += p.thread_group_size)
		for (int j = i + 1; j < p.chromosome_length; j++)
		{
			if (x[i] == x[j])
				clashes++;
			if (abs(i - j) == abs(x[i] - x[j]))
				clashes++;
		}

	// instead of using atomics, we can use a warp level parallel reduction to sum the thread chunks
	for (int offset = p.thread_group_size / 2; offset > 0; offset /= 2)
		clashes += __shfl_down_sync(active_threads_mask, clashes, offset, p.thread_group_size);

	// finally, use thread 0 to update the actual fitness value 
	if (threadIdx.x == 0)
		*fitness = -clashes; // negative in this case, so it becomes a maximisation problem 
}


__device__ void calculate_rosenbrock_fitness(
	void* __restrict__ x1,
	float* __restrict__ fitness,
	const unsigned active_threads_mask,
	const parameters p)
{
	float* x = (float*)x1;
	float temp_fitness = 0.0;
	float dx = 0.0;

	// calculate Tp.SIZE chunks of the function
	for (int i = threadIdx.x; i < p.chromosome_length - 1; i += p.thread_group_size)
	{
		dx = x[i + 1] - x[i] * x[i];
		temp_fitness += 100.0 * dx * dx;
		dx = 1.0 - x[i];
		temp_fitness += dx * dx;
	}

	for (int offset = p.thread_group_size / 2; offset > 0; offset /= 2)
		temp_fitness += __shfl_down_sync(active_threads_mask, temp_fitness, offset, p.thread_group_size);

	// finally, use thread 0 to update the actual fitness value 
	if (threadIdx.x == 0)
		*fitness = -temp_fitness;
}


__device__ void calculate_griewank_fitness(
	void* __restrict__ x1,
	float* __restrict__ fitness,
	const unsigned active_threads_mask,
	const parameters p)
{
	float* x = (float*)x1;
	float sum = 0.0;
	float prod = 1.0;

	// each thread calculates chromosome_length / thread_group chunks 
	for (int i = threadIdx.x; i < p.chromosome_length; i += p.thread_group_size)
	{
		sum += x[i] * x[i];
		prod *= cosf(x[i] / sqrtf(i + 1));
	}

	// use warp shuffle instructions to sum/multiply the 'thread chunks"
	for (int offset = p.thread_group_size / 2; offset > 0; offset /= 2)
	{
		sum += __shfl_down_sync(active_threads_mask, sum, offset, p.thread_group_size);
		prod *= __shfl_down_sync(active_threads_mask, prod, offset, p.thread_group_size);
	}

	// finally thread 0 finishes the calculation 
	if (threadIdx.x == 0)
		*fitness = -((sum / 4000.0) - prod + 1.0);
}


__device__ void calculate_weierstrass_fitness(
	void* __restrict__ x1,
	float* __restrict__ fitness,
	const unsigned active_threads_mask,
	const parameters p)
{
	float* x = (float*)x1;
	float temp_fitness = 0.0;

	// calculate Tp.SIZE chunks of the function
	for (int i = threadIdx.x; i < p.chromosome_length; i += p.thread_group_size)
	{
		for (int j = 0; j < 100; j++)
		{
			temp_fitness += powf(2., -j * 0.25) * sin(powf(2., j) * x[i]);
		}
	}

	for (int offset = p.thread_group_size / 2; offset > 0; offset /= 2)
		temp_fitness += __shfl_down_sync(active_threads_mask, temp_fitness, offset, p.thread_group_size);

	// finally, use thread 0 to update the actual fitness value 
	if (threadIdx.x == 0)
		*fitness = temp_fitness;
}


__device__ void calculate_schwefel_fitness(
	void* __restrict__ x1,
	float* __restrict__ fitness,
	const unsigned active_threads_mask,
	const parameters p)
{
	float* x = (float*)x1;
	float temp_fitness = 0.0;

	// calculate thread_group size chunks of the function
	for (int i = threadIdx.x; i < p.chromosome_length; i += p.thread_group_size)
		temp_fitness += x[i] * sin(sqrt(fabsf(x[i])));
	// now sum the thread_group chunks 
	for (int offset = p.thread_group_size / 2; offset > 0; offset /= 2)
		temp_fitness += __shfl_down_sync(active_threads_mask, temp_fitness, offset, p.thread_group_size);

	// finally, use thread 0 to calculate the inverse final fitness
	if (threadIdx.x == 0)
		*fitness = -(418.9829 * p.chromosome_length - temp_fitness);
}

__device__ void calculate_rastrigin_fitness(
	void* __restrict__ x1,
	float* __restrict__ fitness,
	const unsigned active_threads_mask,
	const parameters p)
{
	float* x = (float*)x1;
	float temp_fitness = 0.0;
	const int a = 10;

	// calculate thread_group size chunks of the function
	for (int i = threadIdx.x; i < p.chromosome_length; i += p.thread_group_size)
		temp_fitness += x[i] * x[i] - a * cosf(2 * M_PI * x[i]);
	// now sum the thread_group chunks 
	for (int offset = p.thread_group_size / 2; offset > 0; offset /= 2)
		temp_fitness += __shfl_down_sync(active_threads_mask, temp_fitness, offset, p.thread_group_size);

	// finally, use thread 0 to calculate the inverse final fitness
	if (threadIdx.x == 0)
		*fitness = -(a * p.chromosome_length + temp_fitness);
}



// this tournament needs to split the population into p.migrants sections and each 
	// section will have a tournament performed in it
	// this prevents multiple TGs replacing the same individual, corrupting it's data. 

__device__ int migrant_tournament_selection(
	const float* __restrict__ fitness,
	curandState* __restrict__ state,
	const unsigned active_threads_mask,
	const parameters p)
{
	int chunk_size = p.island_population_size / p.number_of_migrants;
	int parent = (curand(state) % chunk_size) + threadIdx.y * chunk_size;
	int temp_ind;

	// this mask determines which threads of the warp perform the tournament 
	unsigned tournament_mask = __ballot_sync(active_threads_mask, threadIdx.x < p.tournament_size);

	// this forces thread 0 of the TG to perform the tournament selection
	if (p.thread_group_size < p.tournament_size)
	{
		if (threadIdx.x == 0)
		{
			for (int i = 0; i < p.tournament_size; i++)
			{
				temp_ind = (curand(state) % chunk_size) + threadIdx.y * chunk_size;

				if (!p.maximisation && fitness[parent] < fitness[temp_ind])
					parent = temp_ind;
				else if (p.maximisation && fitness[parent] > fitness[temp_ind])
					parent = temp_ind;
			}
		}
	}

	// otherwise use the TG accelerated tournament selection 
	else if (threadIdx.x < p.tournament_size)
	{
		for (int offset = p.tournament_size; offset > 0; offset /= 2)
		{
			temp_ind = __shfl_down_sync(tournament_mask, parent, offset, p.tournament_size);

			if (!p.maximisation && fitness[parent] < fitness[temp_ind])
				parent = temp_ind;
			else if (p.maximisation && fitness[parent] > fitness[temp_ind])
				parent = temp_ind;
		}
	}
	// broadcast the parent to the whole TG 
	parent = __shfl_sync(active_threads_mask, parent, 0, p.thread_group_size);

	return parent;
}

/*
* Conducts a tournament selection in the calling warp. All calling threads return the winner of the tournament.
* If the number of calling threads exceeds the tournament size, all threads will still return the winner however,
* these threads will be thread blocked during the tournament selection process.
* fitness:	an array of fitness values for the tournament
* tournament_size: the number of individuals participating in the tournament
* maximisation:		boolean value for whether the tournament maximises the fitness, if false will perform a minimisation
* state:	Thread local random state
*/
__device__ int tournament_selection(
	const float* __restrict__ fitness,
	curandState* __restrict__ state,
	const unsigned active_threads_mask,
	const parameters p)
{
	int parent = curand(state) % p.island_population_size;
	int temp_ind;

	// this mask determines which threads of the warp perform the tournament 
	unsigned tournament_mask = __ballot_sync(active_threads_mask, threadIdx.x < p.tournament_size);

	// this forces thread 0 of the TG to perform the tournament selection
	if (p.thread_group_size < p.tournament_size)
	{
		if (threadIdx.x == 0)
		{
			for (int i = 0; i < p.tournament_size; i++)
			{
				temp_ind = curand(state) % p.island_population_size;

				if (p.maximisation && fitness[parent] < fitness[temp_ind])
					parent = temp_ind;
				else if (!p.maximisation && fitness[parent] > fitness[temp_ind])
					parent = temp_ind;
			}
		}
	}

	// otherwise use the TG accelerated tournament selection 
	else if (threadIdx.x < p.tournament_size)
	{
		for (int offset = p.tournament_size / 2; offset > 0; offset /= 2)
		{
			temp_ind = __shfl_down_sync(tournament_mask, parent, offset, p.tournament_size);

			if (p.maximisation && fitness[parent] < fitness[temp_ind])
				parent = temp_ind;
			else if (!p.maximisation && fitness[parent] > fitness[temp_ind])
				parent = temp_ind;
		}
	}
	// broadcast the parent to the whole TG 
	parent = __shfl_sync(active_threads_mask, parent, 0, p.thread_group_size);

	return parent;
}


/*
* Overloaded tournament selection that ensures that the returned parent is not the same as the other_parent argument
* fitness:	an array of fitness values for the tournament
* tournament_size: the number of individuals participating in the tournament
* maximisation:		boolean value for whether the tournament maximises the fitness, if false will perform a minimisation
* state:	Thread local random state
*/
__device__ int tournament_selection(
	const float* __restrict__ fitness,
	const int other_parent,
	curandState* __restrict__ state,
	const unsigned active_threads_mask,
	const parameters p)
{
	int parent = curand(state) % p.island_population_size;
	while (parent == other_parent)
		parent = curand(state) % p.island_population_size;
	int temp_ind;

	// this mask determines which threads of the warp perform the tournament 
	unsigned tournament_mask = __ballot_sync(active_threads_mask, threadIdx.x < p.tournament_size);

	// this forces thread 0 of the TG to perform the tournament selection
	if (p.thread_group_size < p.tournament_size)
	{
		if (threadIdx.x == 0)
		{
			for (int i = 0; i < p.tournament_size; i++)
			{
				temp_ind = curand(state) % p.island_population_size;
				while (temp_ind == other_parent)
					temp_ind = curand(state) % p.island_population_size;

				if (p.maximisation && fitness[parent] < fitness[temp_ind])
					parent = temp_ind;
				else if (!p.maximisation && fitness[parent] > fitness[temp_ind])
					parent = temp_ind;
			}
		}
	}

	// otherwise use the TG accelerated tournament selection 
	else if (threadIdx.x < p.tournament_size)
	{
		for (int offset = p.tournament_size / 2; offset > 0; offset /= 2)
		{
			temp_ind = __shfl_down_sync(tournament_mask, parent, offset, p.tournament_size);

			if (p.maximisation && fitness[parent] < fitness[temp_ind])
				parent = temp_ind;
			else if (!p.maximisation && fitness[parent] > fitness[temp_ind])
				parent = temp_ind;
		}
	}
	parent = __shfl_sync(active_threads_mask, parent, 0, p.thread_group_size);

	return parent;
}


template <typename T>
__device__ void n_segment_crossover(
	const T* __restrict__ population,
	T* __restrict__ offspring,
	const int parent_a,
	const int parent_b,
	curandState* __restrict__ state,
	const parameters p,
	const int n,
	const unsigned active_threads_mask)
{
	int parent;
	int crossover_point;
	int prev_crossover_point = 0;

	for (int i = 0; i < n; i++)
	{
		if (threadIdx.x == 0)
			crossover_point = curand(state) % (p.chromosome_length / n) + i * (p.chromosome_length / n);
		crossover_point = __shfl_sync(active_threads_mask, crossover_point, 0, p.thread_group_size);

		if (i % n == 0)
			parent = parent_a;
		else
			parent = parent_b;

		for (int j = threadIdx.x + prev_crossover_point; j < crossover_point; j += p.thread_group_size)
			offspring[j] = population[parent * p.chromosome_length + j];

		prev_crossover_point = crossover_point;
	}

	if (parent == parent_a)
		parent = parent_b;
	else
		parent = parent_a;

	for (int j = threadIdx.x + prev_crossover_point; j < p.chromosome_length; j += p.thread_group_size)
		offspring[j] = population[parent * p.chromosome_length + j];
}


template <typename T>
__device__ void blx_crossover(
	const T* __restrict__ population,
	T* __restrict__ offspring,
	const int parent_a,
	const int parent_b,
	curandState* __restrict__ state,
	const parameters p)
{
	float alpha = 0.5;
	float x, y, lower, upper, abs_diff, temp;

	for (int i = threadIdx.x; i < p.chromosome_length; i += p.thread_group_size)
	{
		temp = population[parent_a * p.chromosome_length + i] + population[parent_b * p.chromosome_length + i];
		abs_diff = fabsf(population[parent_a * p.chromosome_length + i] - population[parent_b * p.chromosome_length + i]);
		x = 0.5 * (temp - abs_diff);
		y = 0.5 * (temp + abs_diff);
		temp = alpha * (y - x);
		lower = x - temp;
		upper = y + temp;
		offspring[i] = lower + curand_uniform(state) * (upper - lower);
	}
}


template <typename T>
__device__ void arithmetic_crossover(
	const T* __restrict__ population,
	T* __restrict__ offspring,
	const int parent_a,
	const int parent_b,
	curandState* __restrict__ state,
	const parameters p,
	const unsigned mask)
{
	float a;
	if (threadIdx.x == 0)
		a = curand_uniform(state);
	a = __shfl_sync(mask, a, 0, p.thread_group_size);

	for (int i = threadIdx.x; i < p.chromosome_length; i += p.thread_group_size)
		offspring[i] = a * population[parent_a * p.chromosome_length + i] + (1.0 - a) * population[parent_b * p.chromosome_length + i];
}


/*
* Mutates a single gene in the chromosome by selecting a random gene and replacing it with a random value
* between the given min and max values.
* x:		Chromosome to mutate
* min:		minimum possible value
* max:		maximum possible value
* state:	Thread local random state
*/
template<typename T>
__device__ void replacement_mutation(
	T* __restrict__ x,
	curandState* __restrict__ state,
	const parameters p)
{
	if (threadIdx.x == 0 && curand_uniform(state) < p.mutation_probability)
	{
		int rand_ind = curand(state) % p.chromosome_length;
		T rand_val = p.min + curand_uniform(state) * (p.max - p.min);
		x[rand_ind] = rand_val;
	}
}


/*
* Applies a Gaussian mutation to the given chromosome. Each gene in the chromosome
* has gaussian noise scaled by mutation rate added to it. This is used for floating
* point chromosomes.
* x:		Chromosome to mutate
* state:	Thread local random state
* mutation_rate: scaling factor for gaussian noise
*/
template<typename T>
__device__ void gaussian_mutation(
	T* x,
	curandState* state,
	float	mutation_rate,
	unsigned active_threads_mask,
	const parameters p)
{
	float mutation;
	// thread 0 generates crossover 
	if (threadIdx.x == 0)
		mutation = curand_uniform(state);

	mutation = __shfl_sync(active_threads_mask, mutation, 0, p.thread_group_size);

	if (mutation < p.mutation_probability)
		for (int i = threadIdx.x; i < p.chromosome_length; i += p.thread_group_size)
			x[i] += curand_normal(state) * mutation_rate;
}


// overwrites a poor individual from the island with an individual from the neighbouring islands migrant memory 
template <typename T>
__global__ void
L(1024, 1)
genetic_algorithm_migration_kernel(
	T* __restrict__ population,
	float* __restrict__ fitness,
	T* __restrict__ migrants,
	float* __restrict__ migrant_fitness,
	curandState* __restrict__ state,
	const parameters p)
{
	int tid = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	int island_index = threadIdx.y;
	T* island_population = population + blockIdx.x * p.island_population_size * p.chromosome_length;
	float* island_fitness = fitness + blockIdx.x * p.island_population_size;
	int next_island = (blockIdx.x + 1) % p.number_of_islands;
	int migrant = next_island * p.number_of_migrants + island_index;

	// tournament selection to find a poor individual to replace
	int replace = migrant_tournament_selection(island_fitness, &state[tid], FULL_MASK, p);

	__syncthreads();

	for (int i = threadIdx.x; i < p.chromosome_length; i += p.thread_group_size)
		island_population[replace * p.chromosome_length + i] = migrants[migrant * p.chromosome_length + i];

	// update the fitness of the replacement with the migrants fitness value 
	if (threadIdx.x == 0)
		island_fitness[replace] = migrant_fitness[migrant];
}


template <typename T>
__global__ void
L(1024, 1)
genetic_algorithm_kernel(
	const objective_function function,
	T* __restrict__ population,
	float* __restrict__ fitness,
	T* __restrict__ offspring_population,
	T* __restrict__ migrants,
	float* __restrict__ migrant_fitness,
	curandState* __restrict__ state,
	const int generation,
	const int shared_mem_off,
	const int shared_mem_pop,
	const bool migration,
	const parameters p,
	solution* d_solution)
{
	extern __shared__ char smem[];

	float* island_fitness = (float*)smem;
	float* offspring_fitness = (float*)&island_fitness[p.island_population_size];

	T* island_population;
	T* island_offspring;

	if (shared_mem_off)
		island_offspring = (T*)&offspring_fitness[p.island_population_size];
	else
		island_offspring = offspring_population + blockIdx.x * p.island_population_size * p.chromosome_length;
	if (shared_mem_pop)
		island_population = &island_offspring[p.island_population_size * p.chromosome_length];
	else
		island_population = population + blockIdx.x * p.island_population_size * p.chromosome_length;

	// island index is the index of the individual in the island 
	int island_index = threadIdx.y;
	// global index is the index of the individual in the total population
	int global_index = blockIdx.x * p.island_population_size + island_index;
	// tid is the thread ID according to all threads launched with kernel 
	int tid = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	int parent_a;
	int parent_b;

	float mutation_rate = 0.01;
	unsigned active_threads = __ballot_sync(FULL_MASK, global_index < p.total_population_size);

	if (global_index < p.total_population_size)
	{
		if (generation == 0)
		{
			generate_chromosome<T>(island_population + p.chromosome_length * island_index, &state[tid], p);
			__syncwarp(active_threads);

			(*function)(island_population + p.chromosome_length * island_index, &island_fitness[island_index], active_threads, p);
		}

		else
		{
			if (shared_mem_pop)
				for (int i = threadIdx.x; i < p.chromosome_length; i += p.thread_group_size)
					island_population[island_index * p.chromosome_length + i] = population[global_index * p.chromosome_length + i];
			if (threadIdx.x == 0)	// else thread 0 puts the fitness value into shared memory 
				island_fitness[island_index] = fitness[global_index];
		}
	}
	else
		return;

	__syncthreads();

	float crossover;


	for (int i = 1; i <= p.generations; i++)
	{
		if (p.stop_sol_found && d_solution->solution_found > 0)
			return;

		// thread 0 generates crossover 
		if (threadIdx.x == 0)
			crossover = curand_uniform(&state[tid]);

		crossover = __shfl_sync(active_threads, crossover, 0, p.thread_group_size);

		// this mask determines which threads are needed in the crossover 
		unsigned crossover_mask = __ballot_sync(active_threads, crossover < p.crossover_probability);

		if (crossover < p.crossover_probability)
		{
			parent_a = island_index;
			parent_b = tournament_selection(island_fitness, island_index, &state[tid], crossover_mask, p);

			__syncwarp(crossover_mask);

			if (p.combinatorial)
			{
				n_segment_crossover(island_population, island_offspring + island_index * p.chromosome_length, parent_a, parent_b, &state[tid], p, (blockIdx.x % 5) + 1, crossover_mask);
				__syncwarp(crossover_mask);
				replacement_mutation<T>(island_offspring + p.chromosome_length * island_index, &state[tid], p);
			}
			else
			{
				//arithmetic_crossover(island_population, island_offspring + island_index * p.chromosome_length, parent_a, parent_b, &state[tid], p, crossover_mask);
				blx_crossover(island_population, island_offspring + island_index * p.chromosome_length, parent_a, parent_b, &state[tid], p);
				__syncwarp(crossover_mask);
				gaussian_mutation<T>(island_offspring + p.chromosome_length * island_index, &state[tid], mutation_rate, crossover_mask, p);

				// clamp offspring between min and max 
				for (int j = threadIdx.x; j < p.chromosome_length; j += p.thread_group_size)
				{
					island_offspring[p.chromosome_length * island_index + j] = fmaxf(island_offspring[p.chromosome_length * island_index + j], p.min);
					island_offspring[p.chromosome_length * island_index + j] = fminf(island_offspring[p.chromosome_length * island_index + j], p.max);
				}
			}
			__syncwarp(crossover_mask);

			(*function)(island_offspring + p.chromosome_length * island_index, &offspring_fitness[island_index], crossover_mask, p);

			__syncwarp(crossover_mask);

			if (p.elitism)
			{
				// this mask determines which threads are needed in the else condition 
				unsigned elitism_mask = __ballot_sync(crossover_mask, offspring_fitness[island_index] <= island_fitness[island_index]);

				if (offspring_fitness[island_index] > island_fitness[island_index])
				{
					for (int j = threadIdx.x; j < p.chromosome_length; j += p.thread_group_size)
						island_population[p.chromosome_length * island_index + j] = island_offspring[p.chromosome_length * island_index + j];
					if (threadIdx.x == 0)
						island_fitness[island_index] = offspring_fitness[island_index];
				}
				else
				{
					float prob;
					// thread 0 generates crossover 
					if (threadIdx.x == 0)
						prob = curand_uniform(&state[tid]);

					prob = __shfl_sync(elitism_mask, prob, 0, p.thread_group_size);

					if (prob < p.replacement_probability)
					{
						for (int j = threadIdx.x; j < p.chromosome_length; j += p.thread_group_size)
							island_population[p.chromosome_length * island_index + j] = island_offspring[p.chromosome_length * island_index + j];
						if (threadIdx.x == 0)
							island_fitness[island_index] = offspring_fitness[island_index];
					}
				}
			}
			else
			{
				for (int j = threadIdx.x; j < p.chromosome_length; j += p.thread_group_size)
					island_population[p.chromosome_length * island_index + j] = island_offspring[p.chromosome_length * island_index + j];
				if (threadIdx.x == 0)
					island_fitness[island_index] = offspring_fitness[island_index];
			}
		}

		__syncthreads();
		if (threadIdx.x == 0 && fabsf(island_fitness[island_index] - p.solution) <= FLT_EPSILON)
		{
			if (p.stop_sol_found && d_solution->solution_found == 0)
			{
				int val = atomicAdd(&d_solution->solution_found, 1);
				if (val == 0)
				{
					d_solution->solution_generation = (generation + 1) * p.generations + i;
					d_solution->best_fitness = island_fitness[island_index];
					d_solution->solution_index = global_index;
				}
				return;
			}
		}
	}

	unsigned migration_mask = __ballot_sync(active_threads, island_index < p.number_of_migrants);
	// finally copy a good individual from the island into the migrant memory
	// if there are more than 1 island
	if (migration && island_index < p.number_of_migrants)
	{
		int migrant_index = blockIdx.x * p.number_of_migrants + island_index;

		// first we find a good member of our population suitable for migration 
		int good = tournament_selection(island_fitness, &state[tid], migration_mask, p);
		__syncwarp(migration_mask);
		// copy our good member into migrant memory 
		for (int i = threadIdx.x; i < p.chromosome_length; i += p.thread_group_size)
			migrants[migrant_index * p.chromosome_length + i] = island_population[p.chromosome_length * good + i];
		if (threadIdx.x == 0)
			migrant_fitness[migrant_index] = island_fitness[good];
	}

	__syncthreads();

	if (shared_mem_pop)
		for (int i = threadIdx.x; i < p.chromosome_length; i += p.thread_group_size)
			population[global_index * p.chromosome_length + i] = island_population[island_index * p.chromosome_length + i];

	if (threadIdx.x == 0)
		fitness[global_index] = island_fitness[island_index];

	__syncthreads();
}


int analyse_and_record_fitness(
	const float* fitness,
	const int generation,
	std::ofstream& file,
	const parameters p)
{
	// analyse the data 
	int min, max;
	float sum, avg;

	min = 0;
	max = 0;
	sum = 0.0;
	for (int i = 1; i < p.total_population_size; i++)
	{
		sum += fitness[i];
		if (fitness[i] < fitness[min])
			min = i;
		if (fitness[i] > fitness[max])
			max = i;
	}

	avg = sum / p.total_population_size;
	file << generation << "," << fitness[min] << "," << fitness[max] << "," << avg << std::endl;
	return max;
}


void print_parameters(
	const std::string function_name,
	const parameters p,
	const cudaDeviceProp device_prop,
	const bool shared_mem_pop,
	const bool shared_mem_off,
	const int shared_bytes)
{
	std::cout
		<< " ----------------------------------------" << std::endl
		<< "|- CUDA Accelerated Genetic Algorithm" << std::endl
		<< "|- Optimising:                 " << function_name << std::endl
		<< "|- Problem Dimension:          " << p.chromosome_length << std::endl
		<< "|- Individuals per island:     " << p.island_population_size << std::endl
		<< "|- Threads per individual:     " << p.thread_group_size << std::endl
		<< "|- Number of islands:          " << p.number_of_islands << std::endl
		<< "|- Device:                     " << device_prop.name << std::endl
		<< "|- Streaming Multiprocessors:  " << device_prop.multiProcessorCount << std::endl
		<< "|- Shared memory offspring:    " << (shared_mem_off ? "[x]" : "[ ]") << std::endl
		<< "|- Shared memory population:   " << (shared_mem_pop ? "[x]" : "[ ]") << std::endl
		<< "|- Shared memory usage:        " << shared_bytes << "/" << device_prop.sharedMemPerBlock << std::endl
		<< " ----------------------------------------" << std::endl;
}


template<typename T>
void genetic_algorithm(
	const std::string function_name,
	const parameters p,
	const int iteration,
	const unsigned long seed,
	struct solution* h_solution)
{
	// set the CUDA device, in this case the first GPU
	int device = 0;
	cudaDeviceProp device_prop;
	cuda_error_check(cudaGetDeviceProperties(&device_prop, device));
	cuda_error_check(cudaSetDevice(device));

	// determine how much memory we need to store the population and fitness values 
	int population_bytes = p.island_population_size * p.chromosome_length * sizeof(T);
	int fitness_bytes = p.island_population_size * sizeof(float);

	// now we see whether the populations will fit into shared memory 
	bool shared_mem_pop = false;
	bool shared_mem_off = false;
	int shared_bytes = fitness_bytes * 2; // we always want the fitness to be stored in shared memory 
	
	// check whether populationg and offspring will fit 
	if (population_bytes * 2 + shared_bytes < device_prop.sharedMemPerBlockOptin)
	{
		shared_mem_pop = true;
		shared_mem_off = true;
		shared_bytes += population_bytes * 2;
	}
	// check whether only offspring will fit (offspring require fitness evaluation every gen so prioritised)
	else if (population_bytes + shared_bytes < device_prop.sharedMemPerBlockOptin)
	{
		shared_mem_off = true;
		shared_bytes += population_bytes;
	}

	if (shared_bytes >= device_prop.sharedMemPerBlock)
		cudaFuncSetAttribute(genetic_algorithm_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);

	// create launch parameters for CUDA kernels 
	dim3 grid_size(p.number_of_islands);
	dim3 block_size(p.thread_group_size, p.island_population_size);
	dim3 migrant_block_size(p.thread_group_size, p.number_of_migrants);

	// initialise the device random states
	curandState* d_random_states;
	cuda_error_check(cudaMalloc((void**)&d_random_states, p.thread_group_size * p.total_population_size * sizeof(curandState)));

	// call the CUDA kernel, check for errors and synchronise the device 
	initialise_curand_kernel << <grid_size, block_size >> > (d_random_states, seed);
	cuda_error_check(cudaPeekAtLastError());
	cuda_error_check(cudaDeviceSynchronize());

	// allocate host memory
	T* h_population = new T[p.total_population_size * p.chromosome_length];
	T* h_migrants = new T[p.number_of_islands * p.number_of_migrants * p.chromosome_length];
	float* h_fitness = new float[p.total_population_size];
	// allocate device memory 
	T* d_population, * d_offspring, * d_migrants;
	float* d_fitness, * d_migrant_fitness;
	solution* d_solution;

	cuda_error_check(cudaMalloc((void**)&d_population, p.total_population_size * p.chromosome_length * sizeof(T)));
	cuda_error_check(cudaMalloc((void**)&d_offspring, p.total_population_size * p.chromosome_length * sizeof(T)));
	cuda_error_check(cudaMalloc((void**)&d_fitness, p.total_population_size * sizeof(float)));
	cuda_error_check(cudaMalloc((void**)&d_migrants, p.number_of_islands * p.number_of_migrants * p.chromosome_length * sizeof(T)));
	cuda_error_check(cudaMalloc((void**)&d_migrant_fitness, p.number_of_islands * p.number_of_migrants * sizeof(float)));
	cuda_error_check(cudaMalloc((void**)&d_solution, sizeof(solution)));

	// set the device solution struct to all 0's
	cuda_error_check(cudaMemset(d_solution, 0, sizeof(solution)));

	// determine functions 
	// create host side function pointers 
	objective_function h_objective_function;
	// copy function pointers from device to host 	
	if (function_name.compare("n-queens") == 0)
		cudaMemcpyFromSymbol(&h_objective_function, d_calculate_n_queens_fitness, sizeof(objective_function));
	else if (function_name.compare("rosenbrock") == 0)
		cudaMemcpyFromSymbol(&h_objective_function, d_calculate_rosenbrock_fitness, sizeof(objective_function));
	else if (function_name.compare("griewank") == 0)
		cudaMemcpyFromSymbol(&h_objective_function, d_calculate_griewank_fitness, sizeof(objective_function));
	else if (function_name.compare("weierstrass") == 0)
		cudaMemcpyFromSymbol(&h_objective_function, d_calculate_weierstrass_fitness, sizeof(objective_function));
	else if (function_name.compare("schwefel") == 0)
		cudaMemcpyFromSymbol(&h_objective_function, d_calculate_schwefel_fitness, sizeof(objective_function));
	else if (function_name.compare("rastrigin") == 0)
		cudaMemcpyFromSymbol(&h_objective_function, d_calculate_rastrigin_fitness, sizeof(objective_function));
	else
	{
		std::cout << "Unknown function" << std::endl;
		exit(0);
	}
	// remove whitespace from device name for file saving purposes 
	std::string device_name = std::string(device_prop.name);
	std::transform(device_name.begin(), device_name.end(), device_name.begin(), [](char ch) { return ch == ' ' ? '_' : ch; });

	std::ofstream file;

	if (p.log_data)
	{
		// open a file to record fitness data 	
		std::ostringstream oss;
		oss << "data\\" << function_name << "_" << device_name << ".csv";
		if (iteration == 0) // if iteration one, create a new file and dump all the parameters
		{
			//print_parameters(function_name, p, device_prop, shared_mem_pop, shared_mem_off, shared_bytes);

			file.open(oss.str());
			if (!file.is_open())
			{
				std::cout << "Error creating file" << std::endl;
				return;
			}
			file
				<< "chromosome_length," << p.chromosome_length << std::endl
				<< "generations," << p.generations << std::endl
				<< "migrations," << p.migrations << std::endl
				<< "total_generations," << p.total_generations << std::endl
				<< "number_of_islands," << p.number_of_islands << std::endl
				<< "island_population_size," << p.island_population_size << std::endl
				<< "total_population_size," << p.total_population_size << std::endl
				<< "thread_group_size," << p.thread_group_size << std::endl
				<< "tournament_size," << p.tournament_size << std::endl
				<< "number_of_migrants," << p.number_of_migrants << std::endl
				<< "min," << p.min << std::endl
				<< "max," << p.max << std::endl
				<< "mutation_probability," << p.mutation_probability << std::endl
				<< "crossover_probability," << p.crossover_probability << std::endl
				<< "replacement_probability," << p.replacement_probability << std::endl
				<< "solution,elitism," << p.solution << std::endl
				<< "stop_sol_found," << p.stop_sol_found << std::endl
				<< "combinatorial," << p.combinatorial << std::endl
				<< "maximisation," << p.maximisation << std::endl;
		}
		else // else we append to the existing file
			file.open(oss.str(), std::fstream::app);

		if (!file.is_open())
		{
			std::cout << "Error creating file" << std::endl;
			return;
		}

		file << "iteration,generation,min,max,avg" << std::endl;
	}

	// setup cuda timer to time the algorithm
	float time;
	cudaEvent_t start, stop;
	cuda_error_check(cudaEventCreate(&start));
	cuda_error_check(cudaEventCreate(&stop));
	cuda_error_check(cudaEventRecord(start, 0));

	// ----- BEGIN EVOLUTIONARY PROCESS -----

	// run the genetic algorithm kernel 
	genetic_algorithm_kernel<T> << < grid_size, block_size, shared_bytes >> > (
		h_objective_function,
		d_population,
		d_fitness,
		d_offspring,
		d_migrants,
		d_migrant_fitness,
		d_random_states,
		0,
		shared_mem_off,
		shared_mem_pop,
		true ? p.number_of_islands > 1 : false,
		p,
		d_solution);
	cuda_error_check(cudaPeekAtLastError());
	cuda_error_check(cudaDeviceSynchronize());

	// grab the solution struct from device and check if a solution has been found 
	cuda_error_check(cudaMemcpy(h_solution, d_solution, sizeof(solution), cudaMemcpyDeviceToHost));


	if (p.log_data)
	{
		// copy fitness data to host 
		cuda_error_check(cudaMemcpy(h_fitness, d_fitness, p.total_population_size * sizeof(float), cudaMemcpyDeviceToHost));
		// analyse the fitness data 
		int best = analyse_and_record_fitness(h_fitness, p.generations, file, p);
	}

	for (int m = 1; m < p.migrations; m++)
	{
		// run the migration kernel
		if (p.number_of_islands > 1)
		{
			genetic_algorithm_migration_kernel<T> << <grid_size, migrant_block_size >> > (
				d_population,
				d_fitness,
				d_migrants,
				d_migrant_fitness,
				d_random_states,
				p);
			cuda_error_check(cudaPeekAtLastError());
			cuda_error_check(cudaDeviceSynchronize());
		}

		// run the genetic algorithm kernel 
		genetic_algorithm_kernel<T> << <grid_size, block_size, shared_bytes >> > (
			h_objective_function,
			d_population,
			d_fitness,
			d_offspring,
			d_migrants,
			d_migrant_fitness,
			d_random_states,
			m,
			shared_mem_off,
			shared_mem_pop,
			true ? p.number_of_islands > 1 : false,
			p,
			d_solution);

		cuda_error_check(cudaPeekAtLastError());
		cuda_error_check(cudaDeviceSynchronize());

		// grab the solution struct from device and check if a solution has been found 
		cuda_error_check(cudaMemcpy(h_solution, d_solution, sizeof(solution), cudaMemcpyDeviceToHost));
		if (p.stop_sol_found && h_solution->solution_found > 0)
			break;

		if (p.log_data)
		{
			// copy fitness data to host 
			cuda_error_check(cudaMemcpy(h_fitness, d_fitness, p.total_population_size * sizeof(float), cudaMemcpyDeviceToHost));
			// analyse the fitness data 
			int best = analyse_and_record_fitness(h_fitness, p.generations, file, p);
		}
	}

	// copy the population data to host 
	cuda_error_check(cudaMemcpy(h_population, d_population, p.total_population_size * p.chromosome_length * sizeof(T), cudaMemcpyDeviceToHost));
	cuda_error_check(cudaMemcpy(h_fitness, d_fitness, p.total_population_size * sizeof(float), cudaMemcpyDeviceToHost));

	// stop the timer 
	cuda_error_check(cudaEventRecord(stop, 0));
	cuda_error_check(cudaEventSynchronize(stop));
	cuda_error_check(cudaEventElapsedTime(&time, start, stop));
	long long int ms = time;

	h_solution->ms = ms;

	if (h_solution->solution_found == 0)
	{
		// determine the best solution found 
		int best_index = 0;
		for (int i = 1; i < p.total_population_size; i++)
			if (h_fitness[best_index] < h_fitness[i])
				best_index = i;
		h_solution->best_fitness = h_fitness[best_index];
		h_solution->solution_generation = p.total_generations;
	}

	if (p.log_data)
		file << "Elapsed," << ms << std::endl;

	// clean-up 
	if (p.log_data)
		file.close();

	delete[] h_population;
	delete[] h_migrants;
	delete[] h_fitness;

	cudaFree(d_population);
	cudaFree(d_offspring);
	cudaFree(d_migrants);
	cudaFree(d_fitness);
	cudaFree(d_migrant_fitness);
	cudaFree(d_random_states);
	cudaFree(d_solution);
}

template <typename T>
void run_tests(
	const std::string function_name,
	parameters p)
{
	solution h_solution;

	std::vector<int> dimension = { 100, 250, 500 };
	std::vector<int> islands = { 10, 20, 30, 40, 50 };
	std::vector<int> population_size = { 32 };
	std::vector<int> thread_groups = { 4, 8, 16, 32 };

	if (!p.combinatorial)
		dimension = { 25, 50, 100 };

	int iteration = 0;
	int repeats = 25;

	// open a file to record fitness data 	
	std::ostringstream oss;
	oss << "data\\" << function_name << "_data.csv";
	std::ofstream file;
	file.open(oss.str());
	file << function_name << ",,,thread group size" << std::endl;
	// this depends on the parameters we select 
	file << ",,,";
	for (auto t : thread_groups)
		file << t << ",,,,,";
	file << std::endl;
	file << ",,,";
	for (int i = 0; i < thread_groups.size(); i++)
		file << "ms,,gen,,sol,";
	file << std::endl;
	file << "Dimension,Islands,PopSize,";
	for (int i = 0; i < thread_groups.size(); i++)
		file << "mean,stdev,mean,stdev,,";
	file << std::endl;

	std::cout << " __________________________";
	for (auto t : thread_groups)
		std::cout << "_________________________________________";
	std::cout << std::endl;
	std::cout << "|" << std::setw(25) << function_name << "||" << std::setw(21 * thread_groups.size()) << "Thread group size" << std::setw(22 * thread_groups.size()) << "|" << std::endl;
	std::cout << "|                         ||";
	for (auto t : thread_groups)
		std::cout << std::setw(21) << t << std::setw(22) << "|";
	std::cout << std::endl;
	std::cout << "|Dimension|Islands|PopSize||";
	for (auto t : thread_groups)
		std::cout << "     ms     |    gen     |sol|  best sol  |";
	std::cout << std::endl;

	for (auto dim : dimension)
	{
		// problem dimension
		p.chromosome_length = dim;
		if (p.combinatorial)
			p.max = dim;
		for (auto isl : islands)
		{
			// islands 
			p.number_of_islands = isl;
			for (auto pop_size : population_size)
			{
				// population size 
				p.island_population_size = pop_size;
				p.total_population_size = p.island_population_size * p.number_of_islands;
				p.number_of_migrants = p.island_population_size / 10;
				file << dim << "," << isl << "," << pop_size << ",";
				std::cout << "|" << std::setw(9) << dim << "|" << std::setw(7) << isl << "|" << std::setw(7) << pop_size << "||";
				for (auto tg : thread_groups)
				{
					// thread groups 
					p.thread_group_size = tg;

					if (p.island_population_size == 64 && p.thread_group_size > 16 ||
						p.island_population_size == 128 && p.thread_group_size > 8 ||
						p.island_population_size == 256 && p.thread_group_size > 4 ||
						p.island_population_size == 512 && p.thread_group_size > 2 ||
						p.island_population_size == 1024 && p.thread_group_size > 1)
					{
						std::cout << "------------|------------|---|";
						continue;
					}
					// repeats
					srand(0);
					std::vector<int> times;
					std::vector<int> gens;
					double time_mean = 0.0, time_stdev = 0.0;
					double gen_mean = 0.0, gen_stdev = 0.0;
					int sol_found = 0;
					float best_solution;
					for (int r = 0; r < repeats; r++)
					{
						int seed = rand();
						genetic_algorithm<T>(function_name, p, iteration, seed, &h_solution);
						if (h_solution.solution_generation != 0)
							sol_found++;
						times.push_back(h_solution.ms);
						gens.push_back(h_solution.solution_generation);
						if (r == 0)
							best_solution = h_solution.best_fitness;
						else if (best_solution < h_solution.best_fitness)
							best_solution = h_solution.best_fitness;

					}
					// calculate mean 
					for (auto& n : times)
						time_mean += n;
					time_mean /= times.size();
					// calculate std deviation
					std::for_each(times.begin(), times.end(), [&](const double d) {
						time_stdev += (d - time_mean) * (d - time_mean);
						});
					time_stdev = sqrt(time_stdev / (times.size() - 1));

					// calculate mean 
					for (auto& n : gens)
						gen_mean += n;
					gen_mean /= gens.size();
					// calculate std deviation
					std::for_each(gens.begin(), gens.end(), [&](const double d) {
						gen_stdev += (d - gen_mean) * (d - gen_mean);
						});
					gen_stdev = sqrt(gen_stdev / (gens.size()));
					file << (int)time_mean << "," << (int)time_stdev << "," << (int)gen_mean << "," << (int)gen_stdev << "," << gens.size() << ",";
					std::cout << std::setw(6) << (int)time_mean << " " << std::setw(5) << (int)time_stdev << "|" << std::setw(6) << (int)gen_mean << " " << std::setw(5) << (int)gen_stdev << "|" << std::setw(3) << sol_found << "|" << std::setw(12) << abs(best_solution) << "|";
				}
				file << std::endl;
				std::cout << std::endl;
			}
		}
		std::cout << " --------------------------";
		for (auto t : thread_groups)
			std::cout << "------------------------------";
		std::cout << std::endl;
	}
	file.close();
}


int main()
{
	parameters p;

	p.generations = 100;
	p.migrations = 500;
	p.total_generations = p.migrations * p.generations;
	p.tournament_size = 8;
	p.min = 0;
	p.mutation_probability = 1.0;
	p.crossover_probability = 1.0;
	p.replacement_probability = 0.05;
	p.elitism = true;
	p.stop_sol_found = true;
	p.combinatorial = true;
	p.maximisation = true;
	p.solution = 0.0;
	p.log_data = false;

	run_tests<int>("n-queens", p);

	p.generations = 100;
	//p.migrations = 10000;
	p.total_generations = p.migrations * p.generations;
	p.min = -2.048;
	p.max = 2.048;
	p.mutation_probability = 0.05;
	p.crossover_probability = 0.8;
	p.combinatorial = false;
	p.solution = 0.0;

	run_tests<float>("rosenbrock", p);

	//p.min = -2048;
	//p.max = 2048;
	//run_tests<float>("griewank", p);


}
