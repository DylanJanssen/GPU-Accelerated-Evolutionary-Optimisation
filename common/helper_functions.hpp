#ifndef HELPER_FUNCTIONS_H 
#define HELPER_FUNCTIONS_H 

#include <numeric> 
#include <algorithm>
#include <vector> 
#include <string> 
#include <fstream>
#include <sstream>

int int_divide_up(int a, int b)
{
    return (a % b) ? (a / b) + 1 : a / b;
}

template <typename T>
void mean_and_stdev(std::vector<T> values, float &mean, float &stdev)
{
    auto sum = std::accumulate(values.begin(), values.end(), 0.0);
    mean = sum / values.size();
    std::vector<T> diff(values.size());
    std::transform(values.begin(), values.end(), diff.begin(), [mean](auto x)
                   { return x - mean; });
    auto sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    stdev = std::sqrt(sq_sum / values.size());
}

template <typename T> 
float median(std::vector<T> vec)
{
    if (vec.size() % 2 == 0)
    {
        const auto median_it1 = vec.begin() + vec.size() / 2 - 1;
        const auto median_it2 = vec.begin() + vec.size() / 2;
        std::nth_element(vec.begin(), median_it1, vec.end());
        const auto e1 = *median_it1;
        std::nth_element(vec.begin(), median_it2, vec.end());
        const auto e2 = *median_it2;
        return (e1 + e2) / 2;
    }
    else
    {
        const auto median_it = vec.begin() + vec.size() / 2;
        std::nth_element(vec.begin(), median_it, vec.end());
        return *median_it;
    }
}

void log_data(
    const int function_number, 
    const int dim, 
    const int popsize,
    const std::string algorithm_name,  
    const std::vector<float> &best_log, 
    const std::vector<float> &time_log, 
    const std::vector<int> &eval_log)
{
    std::stringstream ss;
    std::string func = (function_number > 9 ? "" : "0") + std::to_string(function_number);
    ss << "output_data/" << algorithm_name << "/F" << func << "_D" << dim << "_P" << popsize << ".txt";
    std::ofstream output(ss.str());
    if (!output.is_open())
    {
        std::cout << "output file did not open" << std::endl;
        exit(0);
    }
    output << algorithm_name << "/F" << func << "_D" << dim << "_P" << popsize << std::endl;
    output << "Fitness ";
    for (const auto &dim : best_log)
        output << dim << " ";
    output << std::endl;
    output << "evaluations ";
    for (const auto &dim : eval_log)
        output << dim << " ";
    output << std::endl;
    output << "time (ms) ";
    for (const auto &dim : time_log)
        output << dim << " ";
    output << std::endl;
    output.close();
}

// overload for island model DE 
void log_data(
    const int function_number, 
    const int dim, 
    const int island_popsize,
    const int islands, 
    const std::string algorithm_name,  
    const std::vector<float> &best_log, 
    const std::vector<float> &time_log, 
    const std::vector<int> &eval_log)
{
    std::stringstream ss;
    std::string func = (function_number > 9 ? "" : "0") + std::to_string(function_number);
    ss << "output_data/" << algorithm_name << "/F" << func << "_D" << dim << "_P" << island_popsize << "_I" << islands << ".txt";
    std::ofstream output(ss.str());
    if (!output.is_open())
    {
        std::cout << "output file did not open" << std::endl;
        exit(0);
    }
    output << algorithm_name << "/F" << func << "_D" << dim << "_P" << island_popsize << "_I" << islands << std::endl;
    output << "Fitness ";
    for (const auto &dim : best_log)
        output << dim << " ";
    output << std::endl;
    output << "evaluations ";
    for (const auto &dim : eval_log)
        output << dim << " ";
    output << std::endl;
    output << "time (s) ";
    for (const auto &dim : time_log)
        output << dim << " ";
    output << std::endl;
    output.close();
}


#endif 