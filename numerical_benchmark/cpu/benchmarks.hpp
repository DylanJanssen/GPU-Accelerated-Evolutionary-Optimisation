#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include "benchmark_data.hpp"
#include "basic_functions.hpp"
#include "hybrid_functions.hpp"
#include "composition_functions.hpp"


class Benchmark
{
private:
    BenchmarkData data;
    int dim;
    int function;
    float *y;
    float *z;

public:
    Benchmark(int function, int dim)
        : function(function), dim(dim), data(BenchmarkData(function, dim)), y(new float[dim]), z(new float[dim]) {}
    void evaluate(float *x, float *fitness, int popsize);
};

void Benchmark::evaluate(float *x, float *fitness, int popsize)
{
    for (int i = 0; i < popsize; i++)
    {
        switch (function)
        {
        case 1:
            zakharov_function(&x[i * dim], y, z, &fitness[i], dim, data.get_shift_ptr(), data.get_rotate_ptr(), true, true);
            break;
        case 2:
            rosenbrock_function(&x[i * dim], y, z, &fitness[i], dim, data.get_shift_ptr(), data.get_rotate_ptr(), true, true);
            break;
        case 3:
            rastrigin_function(&x[i * dim], y, z, &fitness[i], dim, data.get_shift_ptr(), data.get_rotate_ptr(), true, true);
            break;
        case 4:
            schwefel_function(&x[i * dim], y, z, &fitness[i], dim, data.get_shift_ptr(), data.get_rotate_ptr(), true, true);
            break;
        case 5:
            hybrid_function_1(&x[i * dim], y, z, &fitness[i], dim, data.get_shift_ptr(), data.get_rotate_ptr(), data.get_shuffle_ptr(), true, true);
            break;
        case 6:
            hybrid_function_2(&x[i * dim], y, z, &fitness[i], dim, data.get_shift_ptr(), data.get_rotate_ptr(), data.get_shuffle_ptr(), true, true);
            break;
        case 7:
            hybrid_function_3(&x[i * dim], y, z, &fitness[i], dim, data.get_shift_ptr(), data.get_rotate_ptr(), data.get_shuffle_ptr(), true, true);
            break;
        case 8:
            composition_function_1(&x[i * dim], y, z, &fitness[i], dim, data.get_shift_ptr(), data.get_rotate_ptr(), true);
            break;
        case 9:
            composition_function_2(&x[i * dim], y, z, &fitness[i], dim, data.get_shift_ptr(), data.get_rotate_ptr(), true);
            break;
        case 10:
            composition_function_3(&x[i * dim], y, z, &fitness[i], dim, data.get_shift_ptr(), data.get_rotate_ptr(), true);
            break;
        default:
            printf("\nError: There are only 10 test functions in this test suite!\n");
            *fitness = 0.0;
            break;
        }
    }
}

#endif