#ifndef HYBRID_FUNCTIONS_H
#define HYRBID_FUNCTIONS_H 

#include "basic_functions.hpp"
#include "util.hpp" 

// function prototypes 
void hybrid_function_1(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, int *shuffle_data, bool shift_flag, bool rotate_flag); /* Hybrid Function 1 */
void hybrid_function_2(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, int *shuffle_data, bool shift_flag, bool rotate_flag); /* Hybrid Function 2 */
void hybrid_function_3(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, int *shuffle_data, bool shift_flag, bool rotate_flag); /* Hybrid Function 3 */


void hybrid_function_1(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    int *shuffle_data, 
    bool shift_flag, 
    bool rotate_flag)
{
    scale_shift_and_rotate(x, y, z, dim, shift_data, rotate_data, 1.0f, shift_flag, rotate_flag); /* shift and rotate */
    shuffle(z, y, dim, shuffle_data); 

    const float percentages[] = {0.3f, 0.3f, 0.4f};
    float s = 0.0f;

    int offset = 0;
    int size = ceilf(percentages[0] * dim);
    bent_cigar_function(y + offset, y, z, fitness, size, shift_data, rotate_data, false, false);
    s += *fitness;
    offset += size;
    size = ceilf(percentages[1] * dim);
    hgbat_function(y + offset, y, z, fitness, size, shift_data, rotate_data, false, false);
    s += *fitness;
    offset += size;
    size = dim - offset;
    rastrigin_function(y + offset, y, z, fitness, size, shift_data, rotate_data, false, false);
    *fitness += s;
}



void hybrid_function_2(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    int *shuffle_data, 
    bool shift_flag, 
    bool rotate_flag)
{
    scale_shift_and_rotate(x, y, z, dim, shift_data, rotate_data, 1.0f, shift_flag, rotate_flag); /* shift and rotate */
    shuffle(z, y, dim, shuffle_data);

    const float percentages[] = {0.2f, 0.2f, 0.3f, 0.3f};
    float s = 0.0f;

    int offset = 0;
    int size = ceilf(percentages[0] * dim);
    expanded_schaffer_F6_function(y + offset, y, z, fitness, size, shift_data, rotate_data, false, false);
    s += *fitness;

    offset += size;
    size = ceilf(percentages[1] * dim);
    hgbat_function(y + offset, y, z, fitness, size, shift_data, rotate_data, false, false);
    s += *fitness;

    offset += size;
    size = ceilf(percentages[2] * dim);
    rosenbrock_function(y + offset, y, z, fitness, size, shift_data, rotate_data, false, false);
    s += *fitness;

    offset += size;
    size = dim - offset;
    schwefel_function(y + offset, y, z, fitness, size, shift_data, rotate_data, false, false);
    *fitness += s;
}



void hybrid_function_3(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    int *shuffle_data, 
    bool shift_flag, 
    bool rotate_flag)
{
    scale_shift_and_rotate(x, y, z, dim, shift_data, rotate_data, 1.0f, shift_flag, rotate_flag); /* shift and rotate */
    shuffle(z, y, dim, shuffle_data);
    const float percentages[] = {0.3f, 0.2f, 0.2f, 0.1f, 0.2f};
    float s = 0.0f;

    int offset = 0;
    int size = ceilf(percentages[0] * dim);
    katsuura_function(y + offset, y, z, fitness, size, shift_data, rotate_data, false, false);
    s += *fitness;

    offset += size;
    size = ceilf(percentages[1] * dim);
    happycat_function(y + offset, y, z, fitness, size, shift_data, rotate_data, false, false);
    s += *fitness;

    offset += size;
    size = ceilf(percentages[2] * dim);
    expanded_griewank_plus_rosenbrock_function(y + offset, y, z, fitness, size, shift_data, rotate_data, false, false);
    s += *fitness;

    offset += size;
    size = ceilf(percentages[3] * dim);
    schwefel_function(y + offset, y, z, fitness, size, shift_data, rotate_data, false, false);
    s += *fitness;

    offset += size;
    size = dim - offset;
    ackley_function(y + offset, y, z, fitness, size, shift_data, rotate_data, false, false);
    *fitness += s;
}

#endif 