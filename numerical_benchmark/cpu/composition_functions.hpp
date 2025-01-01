#ifndef COMPOSITION_FUNCTIONS_H 
#define COMPOSITION_FUNCTIONS_H 

#include "basic_functions.hpp"
#include "util.hpp"

// function prototypes 
void composition_function_1(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool rotate_flag); /* Composition Function 1 */
void composition_function_2(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool rotate_flag); /* Composition Function 2 */
void composition_function_3(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool rotate_flag); /* Composition Function 3 */
void composition_calculation(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *delta, float *bias, float *fitness_values, int cf_num);


void composition_function_1(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    bool rotate_flag)
{
    int i, cf_num = 3;
    float fitness_values[3];
    float delta[3] = {10.0f, 20.0f, 30.0f};
    float heights[] = {1.0f, 10.0f, 1.0f};
    float bias[3] = {0.0f, 100.0f, 200.0f};

    i = 0;
    rastrigin_function(x, y, z, &fitness_values[i], dim, &shift_data[i * dim], &rotate_data[i * dim * dim], 1, rotate_flag);
    i = 1;
    griewank_function(x, y, z, &fitness_values[i], dim, &shift_data[i * dim], &rotate_data[i * dim * dim], 1, rotate_flag);
    i = 2;
    schwefel_function(x, y, z, &fitness_values[i], dim, &shift_data[i * dim], &rotate_data[i * dim * dim], 1, rotate_flag);
    for (int i = 0; i < cf_num; i++)
        fitness_values[i] *= heights[i];
    composition_calculation(x, y, z, fitness, dim, shift_data, delta, bias, fitness_values, cf_num);
}

void composition_function_2(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool rotate_flag)
{
    int i, cf_num = 4;
    float fitness_values[4];
    float delta[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    const float heights[] = {10.0f, 1e-6f, 10.0f, 1.0f};
    float bias[4] = {0.0f, 100.0f, 200.0f, 300.0f};

    i = 0;
    ackley_function(x, y, z, &fitness_values[i], dim, &shift_data[i * dim], &rotate_data[i * dim * dim], 1, rotate_flag);
    i = 1;
    high_conditional_elliptic_function(x, y, z, &fitness_values[i], dim, &shift_data[i * dim], &rotate_data[i * dim * dim], 1, rotate_flag);
    i = 2;
    griewank_function(x, y, z, &fitness_values[i], dim, &shift_data[i * dim], &rotate_data[i * dim * dim], 1, rotate_flag);
    i = 3;
    rastrigin_function(x, y, z, &fitness_values[i], dim, &shift_data[i * dim], &rotate_data[i * dim * dim], 1, rotate_flag);
    for (int i = 0; i < cf_num; i++)
        fitness_values[i] *= heights[i];
    composition_calculation(x, y, z, fitness, dim, shift_data, delta, bias, fitness_values, cf_num);
}

void composition_function_3(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool rotate_flag)
{
    int i, cf_num = 5;
    float fitness_values[5];
    float delta[5] = {10.0f, 20.0f, 20.0f, 30.0f, 40.0f};
    const float heights[] = {0.0005f, 1.0f, 10.0f, 1.0f, 10.0f};
    float bias[5] = {0.0f, 100.0f, 200.0f, 300.0f, 400.0f};
    i = 0;
    expanded_schaffer_F6_function(x, y, z, &fitness_values[i], dim, &shift_data[i * dim], &rotate_data[i * dim * dim], 1, rotate_flag);
    i = 1;
    schwefel_function(x, y, z, &fitness_values[i], dim, &shift_data[i * dim], &rotate_data[i * dim * dim], 1, rotate_flag);
    i = 2;
    griewank_function(x, y, z, &fitness_values[i], dim, &shift_data[i * dim], &rotate_data[i * dim * dim], 1, rotate_flag);
    i = 3;
    rosenbrock_function(x, y, z, &fitness_values[i], dim, &shift_data[i * dim], &rotate_data[i * dim * dim], 1, rotate_flag);
    i = 4;
    rastrigin_function(x, y, z, &fitness_values[i], dim, &shift_data[i * dim], &rotate_data[i * dim * dim], 1, rotate_flag);
    for (int i = 0; i < cf_num; i++)
        fitness_values[i] *= heights[i];
    composition_calculation(x, y, z, fitness, dim, shift_data, delta, bias, fitness_values, cf_num);
}

void composition_calculation(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *delta, 
    float *bias, 
    float *fitness_values, 
    int cf_num)
{
    float temp;
    float sq_sum;
    float *weight = z;
    float weight_sum = 0.0f;

    for (int i = 0; i < cf_num; i++)
    {
        sq_sum = 0.0f;
        for (int j = 0; j < dim; j++)
        {
            temp = x[j] - shift_data[i * dim + j];
            sq_sum += temp * temp;
        }
        fitness_values[i] += bias[i];
        weight[i] = (1.0f / sqrt(sq_sum)) * exp(-sq_sum / (2.0 * dim * delta[i] * delta[i]));
        weight_sum += weight[i];
    }
    float s = 0.0f;
    for (int i = 0; i < cf_num; i++)
        s += weight[i] / weight_sum * fitness_values[i];
    *fitness = s;
}

#endif 