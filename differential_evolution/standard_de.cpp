#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <numeric>
#include <vector>
#include <chrono>

#include "../common/helper_functions.hpp"
#include "../numerical_benchmark/cpu/benchmarks.hpp"
#include "cpu/standard_de.hpp"

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

    std::cout << "Running standard de with popsize=" << popsize << std::endl;

    const float lower_bound = -100.0f;
    const float upper_bound = 100.0f;
    const float F = 0.5;
    const float CR = 0.3;
    srand(0);

    Benchmark benchmark(function_number, dim);
    std::vector<float> best_log;
    std::vector<float> time_log;
    std::vector<int> eval_log;

    for (int runs = 1; runs <= total_runs; runs++)
    {
        DifferentialEvolution<float> de(popsize, dim, CR, F, lower_bound, upper_bound);
        auto start = std::chrono::steady_clock::now();
        de.initialise_population();
        benchmark.evaluate(de.get_population_ptr(), de.get_fitness_ptr(), popsize);
        auto best = de.get_best();
        bool solution_found = false;
        if (best < 1e-8)
            solution_found = true;
        int evals = popsize;
        while (evals < max_evaluations && !solution_found)
        {
            de.generate_offspring();
            benchmark.evaluate(de.get_offspring_ptr(), de.get_offspring_fitness_ptr(), popsize);
            best = de.get_best();
            if (best < 1e-8)
                solution_found = true;
            de.replacement();
            evals += popsize;
        }
        auto end = std::chrono::steady_clock::now();
        best = de.get_best();
        std::cout << "Function: " << function_number << " Best fitness: " << best << std::endl;
        best_log.push_back(best);
        float time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0;
        std::cout << "Function: " << function_number << " Best fitness: " << best << " Time: " << time << std::endl;
        time_log.push_back(time);
        eval_log.push_back(evals);
    }

    log_data(function_number, dim, popsize, "standard_de", best_log, time_log, eval_log); 

    auto min_fitness = *std::min_element(best_log.begin(), best_log.end());
    auto max_fitness = *std::max_element(best_log.begin(), best_log.end());
    
    float mean, stdev;
    mean_and_stdev(best_log, mean, stdev);
    float time_mean, time_stdev;
    mean_and_stdev(time_log, time_mean, time_stdev);
    float gen_mean, gen_stdev;
    mean_and_stdev(eval_log, gen_mean, gen_stdev);

    std::cout << function_number << ", " << min_fitness << ", " << max_fitness << ", " << mean
              << ", " << stdev << ", " << time_mean << ", " << time_stdev << ", "
              << gen_mean << ", " << gen_stdev << std::endl;
}
