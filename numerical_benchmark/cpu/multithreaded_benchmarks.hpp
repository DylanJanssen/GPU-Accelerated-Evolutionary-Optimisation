#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include "benchmark_data.hpp"
#include "thread_pool.hpp"
#include "basic_functions.hpp"
#include "hybrid_functions.hpp"
#include "composition_functions.hpp"

class MultithreadedBenchmark
{
private:
    BenchmarkData data;
    ThreadPool thread_pool;
    vector<int> section_starts{0};
    int dim;
    int function;
    int popsize;
    float *y;
    float *z;
    static void evaluate(int function, float *x, float *y, float *z, float *fitness, int dim, int popsize, float *shift_data, float *rotate_data, int *shuffle_data);

public:
    MultithreadedBenchmark(int function, int dim, int popsize);
    void multithreaded_evaluate(float *x, float *fitness, int popsize);
};

MultithreadedBenchmark::MultithreadedBenchmark(int function, int dim, int popsize)
    : function(function), dim(dim), popsize(popsize), data(BenchmarkData(function, dim)), y(new float[dim * popsize]), z(new float[dim * popsize])
{
    // determine how many individuals each thread will process
    const int threads = thread_pool.get_num_threads();
    int section_size = popsize / threads;
    for (int i = 1; i < threads; i++)
    {
        section_starts.push_back(section_starts.back() + section_size);
        if (i <= popsize % threads)
            ;
        section_starts[i]++;
    }
    section_starts.push_back(popsize);
}

void MultithreadedBenchmark::evaluate(
    int function, 
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    int popsize, 
    float *shift_data, 
    float *rotate_data, 
    int *shuffle_data)
{
    for (int i = 0; i < popsize; i++)
    {
        switch (function)
        {
        case 1:
            zakharov_function(&x[i * dim], &y[i * dim], &z[i * dim], &fitness[i], dim, shift_data, rotate_data, true, true);
            break;
        case 2:
            rosenbrock_function(&x[i * dim], &y[i * dim], &z[i * dim], &fitness[i], dim, shift_data, rotate_data, true, true);
            break;
        case 3:
            rastrigin_function(&x[i * dim], &y[i * dim], &z[i * dim], &fitness[i], dim, shift_data, rotate_data, true, true);
            break;
        case 4:
            schwefel_function(&x[i * dim], &y[i * dim], &z[i * dim], &fitness[i], dim, shift_data, rotate_data, true, true);
            break;
        case 5:
            hybrid_function_1(&x[i * dim], &y[i * dim], &z[i * dim], &fitness[i], dim, shift_data, rotate_data, shuffle_data, true, true);
            break;
        case 6:
            hybrid_function_2(&x[i * dim], &y[i * dim], &z[i * dim], &fitness[i], dim, shift_data, rotate_data, shuffle_data, true, true);
            break;
        case 7:
            hybrid_function_3(&x[i * dim], &y[i * dim], &z[i * dim], &fitness[i], dim, shift_data, rotate_data, shuffle_data, true, true);
            break;
        case 8:
            composition_function_1(&x[i * dim], &y[i * dim], &z[i * dim], &fitness[i], dim, shift_data, rotate_data, true);
            break;
        case 9:
            composition_function_2(&x[i * dim], &y[i * dim], &z[i * dim], &fitness[i], dim, shift_data, rotate_data, true);
            break;
        case 10:
            composition_function_3(&x[i * dim], &y[i * dim], &z[i * dim], &fitness[i], dim, shift_data, rotate_data, true);
            break;
        default:
            printf("\nError: There are only 10 test functions in this test suite!\n");
            *fitness = 0.0;
            break;
        }
    }
}

void MultithreadedBenchmark::multithreaded_evaluate(float *x, float *fitness, int popsize)
{
    for (int i = 0; i < section_starts.size() - 1; i++)
    {
        int offset = section_starts[i];
        int ptr_offset = offset * dim;
        int size = section_starts[i + 1] - section_starts[i];

        thread_pool.submit_job(&MultithreadedBenchmark::evaluate, function,
                               x + ptr_offset,
                               y + ptr_offset,
                               z + ptr_offset,
                               fitness + offset,
                               dim,
                               size,
                               data.get_shift_ptr(),
                               data.get_rotate_ptr(),
                               data.get_shuffle_ptr());
    }
    thread_pool.wait_for_jobs();
}

#endif