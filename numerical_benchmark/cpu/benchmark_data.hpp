#ifndef BENCHMARK_DATA_H
#define BENCHMARK_DATA_H

#include <unordered_map> 
#include <fstream>       
#include <sstream>       
#include <iostream>      
#include <string>        
#include <cstdlib>       

using namespace std;

// BenchmarkData class is used to store benchmark data
class BenchmarkData
{
protected:
    int dim;
    int function;
    float *shift_data = nullptr;
    float *rotate_data = nullptr;
    int *shuffle_data = nullptr;
    unordered_map<int, int> composition_functions{{8, 3}, {9, 4}, {10, 5}};
    int hybrid_start = 5;
    int composition_start = 8;

public:
    BenchmarkData(int function, int dim);
    ~BenchmarkData();
    int get_function() { return function; }
    int get_dim() { return dim; }
    int get_cf_num() { return composition_functions.at(function); }
    float *get_shift_ptr() { return shift_data; }
    float *get_rotate_ptr() { return rotate_data; }
    int *get_shuffle_ptr() { return shuffle_data; }
};

BenchmarkData::BenchmarkData(int function, int dim) : function(function), dim(dim)
{
    if (!(dim == 2 || dim == 50 || dim == 100))
    {
        std::cout << "Benchmarks are defined for dim=2, dim=50, and dim=100" << std::endl;
        exit(1);
    }
    auto cf_num = 1;
    if (function >= composition_start) // composition functions
        cf_num = composition_functions.at(function);

    // rotation data
    std::stringstream ss;
    ss << "numerical_benchmark/benchmark_data/M_" << function << "_D" << dim << ".txt";
    std::ifstream rotation_file(ss.str());
    if (!rotation_file.is_open())
    {
        std::cout << "Failed to open: " << ss.str() << std::endl;
        exit(1);
    }
    rotate_data = new float[dim * dim * cf_num];
    for (int i = 0; i < dim * dim * cf_num; i++)
        rotation_file >> rotate_data[i];
    rotation_file.close();

    // shift data
    stringstream().swap(ss); // swap ss with default constructed stringstream
    ss << "numerical_benchmark/benchmark_data/shift_data_" << function << ".txt";
    std::ifstream shift_file(ss.str());
    if (!shift_file.is_open())
    {
        std::cout << "Failed to open: " << ss.str() << std::endl;
        exit(1);
    }
    shift_data = new float[dim * cf_num];
    for (int i = 0; i < dim * cf_num; i++)
        shift_file >> shift_data[i];
    shift_file.close();

    // hybrid function shuffle data
    if (function >= hybrid_start && function <= composition_start)
    {
        stringstream().swap(ss); // swap ss with default constructed stringstream
        ss << "numerical_benchmark/benchmark_data/shuffle_data_" << function << "_D" << dim << ".txt";
        std::ifstream shuffle_file(ss.str());
        if (!shuffle_file.is_open())
        {
            std::cout << "Failed to open: " << ss.str() << std::endl;
            exit(1);
        }
        shuffle_data = new int[dim];
        for (int i = 0; i < dim; i++)
            shuffle_file >> shuffle_data[i];
        shift_file.close();
    }
}

BenchmarkData::~BenchmarkData()
{
    if (shift_data != nullptr)
        delete[] shift_data;
    if (rotate_data != nullptr)
        delete[] rotate_data;
    if (shuffle_data != nullptr)
        delete[] shuffle_data;
}

#endif