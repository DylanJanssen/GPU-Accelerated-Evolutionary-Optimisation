#ifndef BASIC_FUNCTIONS_H
#define BASIC_FUNCTIONS_H 

#include "util.hpp"
#include <cmath>

#define E 2.7182818284590452353602874713526625

// function prototypes 
void zakharov_function(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool shift_flag, bool rotate_flag);                          /* ZAKHAROV */
void rosenbrock_function(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool shift_flag, bool rotate_flag);                        /* Rosenbrock'shuffle_data */
void rastrigin_function(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool shift_flag, bool rotate_flag);                         /* Rastrigin'shuffle_data  */
void schwefel_function(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool shift_flag, bool rotate_flag);                          /* Schwefel'shuffle_data */
void bent_cigar_function(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool shift_flag, bool rotate_flag);                        /* Discus */
void hgbat_function(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool shift_flag, bool rotate_flag);                                 /* HGBat  */
void katsuura_function(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool shift_flag, bool rotate_flag);                          /* Katsuura */
void happycat_function(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool shift_flag, bool rotate_flag);                          /* HappyCat */
void expanded_griewank_plus_rosenbrock_function(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool shift_flag, bool rotate_flag); /* Griewank-Rosenbrock  */
void ackley_function(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool shift_flag, bool rotate_flag);                            /* Ackley'shuffle_data */
void high_conditional_elliptic_function(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool shift_flag, bool rotate_flag);         /* Ellipsoidal */
void griewank_function(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool shift_flag, bool rotate_flag);                          /* Griewank'shuffle_data  */
void expanded_schaffer_F6_function(float *x, float *y, float *z, float *fitness, int dim, float *shift_data, float *rotate_data, bool shift_flag, bool rotate_flag);              /* Expanded Scaffer¡¯s F6  */


void zakharov_function(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    bool shift_flag, 
    bool rotate_flag)
{
    scale_shift_and_rotate(x, y, z, dim, shift_data, rotate_data, 1.0f, shift_flag, rotate_flag); // shift and rotate
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = 0; i < dim; i++)
    {
        sum1 += z[i] * z[i];
        sum2 += 0.5f * (i + 1) * z[i];
    }
    *fitness = sum1 + sum2 * sum2 + powf(sum2, 4);
}

void rosenbrock_function(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    bool shift_flag, 
    bool rotate_flag)
{
    scale_shift_and_rotate(x, y, z, dim, shift_data, rotate_data, 0.02048f, shift_flag, rotate_flag); /* shift and rotate */
    // shift to origin 
    for (int i = 0; i < dim; i++)
        z[i] += 1.0f; 
    float temp1, temp2;
    *fitness = 0.0;
    for (int i = 0; i < dim - 1; i++)
    {
        temp1 = z[i] * z[i] - z[i + 1];
        temp2 = z[i] - 1.0f;
        *fitness += 100.0f * temp1 * temp1 + temp2 * temp2;
    }
}

void rastrigin_function(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    bool shift_flag, 
    bool rotate_flag)
{
    scale_shift_and_rotate(x, y, z, dim, shift_data, rotate_data, 0.0512f, shift_flag, rotate_flag); /* shift and rotate */
    *fitness = 0.0f; 
    for (int i = 0; i < dim; i++)
        *fitness += (z[i] * z[i] - 10.0f * cos(2.0 * M_PI * z[i]) + 10.0f);
}

void schwefel_function(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    bool shift_flag, 
    bool rotate_flag)
{
    scale_shift_and_rotate(x, y, z, dim, shift_data, rotate_data, 10.0f, shift_flag, rotate_flag); /* shift and rotate */
    float s=0.0f, tmp;

    for (int i = 0; i < dim; i++)
    {
        z[i] += 420.9687462275036f;
        if (z[i] > 500.0f)
        {
            s -= (500.0f - fmodf(z[i], 500.0f)) * sinf(powf(500.0f - fmodf(z[i], 500.0f), 0.5f));
            tmp = (z[i] - 500.0f) / 100.0f;
            s += tmp * tmp / dim;
        }
        else if (z[i] < -500.0f)
        {
            s -= (-500.0f + fmodf(fabsf(z[i]), 500.0f)) * sinf(powf(500.0f - fmodf(fabsf(z[i]), 500.0f), 0.5f));
            tmp = (z[i] + 500.0f) / 100.0f;
            s += tmp * tmp / dim;
        }
        else
            s -= z[i] * sinf(powf(fabsf(z[i]), 0.5f));
    }
    *fitness = s + 418.9828872724338f * dim;
}

void bent_cigar_function(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    bool shift_flag, 
    bool rotate_flag)
{
    scale_shift_and_rotate(x, y, z, dim, shift_data, rotate_data, 1.0f, shift_flag, rotate_flag); /* shift and rotate */
    float s = z[0] * z[0];
    for (int i = 1; i < dim; i++)
        s += 1000000.0f * z[i] * z[i];
    *fitness = s;
}

void hgbat_function(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    bool shift_flag, 
    bool rotate_flag)
{
    scale_shift_and_rotate(x, y, z, dim, shift_data, rotate_data, 0.05f, shift_flag, rotate_flag); /* shift and rotate */
    float s = 0.0f, alpha=0.25f, r2=0.0f;
 
    for (int i = 0; i < dim; i++)
    {
        z[i] = z[i] - 1.0f; // shift to orgin
        r2 += z[i] * z[i];
        s += z[i];
    }
    *fitness = powf(fabsf(r2 * r2 - s * s), 2 * alpha) + (0.5f * r2 + s) / dim + 0.5f;
}


void expanded_schaffer_F6_function(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    bool shift_flag, 
    bool rotate_flag)
{
    scale_shift_and_rotate(x, y, z, dim, shift_data, rotate_data, 1.0f, shift_flag, rotate_flag); /* shift and rotate */
    float s=0.0f, temp1, temp2, a;
    for (int i = 0; i < dim; i++)
    {
        a = z[i] * z[i] + z[(i+1)%dim] * z[(i+1)%dim]; 
        temp1 = sinf(sqrtf(a));
        temp1 *= temp1;
        temp2 = 1.0f + 0.001f * a;
        s += 0.5f + (temp1 - 0.5f) / (temp2 * temp2);
    }
    *fitness = s;
}


void katsuura_function(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    bool shift_flag, 
    bool rotate_flag)
{
    scale_shift_and_rotate(x, y, z, dim, shift_data, rotate_data, 0.05f, shift_flag, rotate_flag); /* shift and rotate */
    float s = 1.0f;
    float temp1, temp2, temp3, temp4;
    temp4 = powf(1.0f * dim, 1.2f);
    for (int i = 0; i < dim; i++)
    {
        temp1 = 0.0f;
        for (int j = 1; j <= 32; j++)
        {
            temp2 = powf(2.0f, j);
            temp3 = temp2 * z[i];
            temp1 += fabsf(temp3 - floorf(temp3 + 0.5f)) / temp2;
        }
        s *= powf(1.0f + (i + 1) * temp1, 10.0f / temp4);
    }
    temp2 = 10.0f / dim / dim; 
    *fitness = s * temp2 - temp2;
}

void happycat_function(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    bool shift_flag, 
    bool rotate_flag)
{
    scale_shift_and_rotate(x, y, z, dim, shift_data, rotate_data, 0.05f, shift_flag, rotate_flag); /* shift and rotate */
    float s = 0.0f, r = 0.0f, alpha = 0.125f;   

    for (int i = 0; i < dim; i++)
    {
        z[i] -= 1.0f; // shift to orgin
        r += z[i] * z[i];
        s += z[i];
    }
    *fitness = powf(fabsf(r - dim), 2 * alpha) + (0.5f * r + s) / dim + 0.5f;
}

void expanded_griewank_plus_rosenbrock_function(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    bool shift_flag, 
    bool rotate_flag)
{
    scale_shift_and_rotate(x, y, z, dim, shift_data, rotate_data, 0.05f, shift_flag, rotate_flag); /* shift and rotate */
    float s = 0.0f;
    float temp1, temp2, temp3;
    for (int i = 0; i < dim; i++) // shift to origin 
        z[i] += 1.0f; 
    for (int i = 0; i < dim; i++)
    {
        // rosenbrock 
        temp1 = z[i] * z[i] - z[(i + 1) % dim];
        temp2 = z[i] - 1.0f;
        temp3 = 100.0f * temp1 * temp1 + temp2 * temp2;
        // end rosenbrock
        // griewank
        s += (temp3 * temp3) / 4000.0f - cosf(temp3) + 1.0f;
    }
    *fitness = s;
}

void ackley_function(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    bool shift_flag, 
    bool rotate_flag)
{
    scale_shift_and_rotate(x, y, z, dim, shift_data, rotate_data, 1.0f, shift_flag, rotate_flag); /* shift and rotate */
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = 0; i < dim; i++)
    {
        sum1 += z[i] * z[i];
        sum2 += cosf(2.0f * M_PI * z[i]);
    }
    sum1 = -0.2f * sqrtf(sum1 / dim);
    sum2 /= dim;
    *fitness = E - 20.0f * expf(sum1) - expf(sum2) + 20.0f;
}


void griewank_function(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    bool shift_flag, 
    bool rotate_flag)
{
    scale_shift_and_rotate(x, y, z, dim, shift_data, rotate_data, 6.0f, shift_flag, rotate_flag); /* shift and rotate */
    float s = 0.0, p = 1.0;

    for (int i = 0; i < dim; i++)
    {
        s += z[i] * z[i];
        p *= cosf(z[i] / sqrtf(1.0f + i));
    }
    *fitness = 1.0f + s / 4000.0f - p;
}

void high_conditional_elliptic_function(
    float *x, 
    float *y, 
    float *z, 
    float *fitness, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    bool shift_flag, 
    bool rotate_flag)
{
    scale_shift_and_rotate(x, y, z, dim, shift_data, rotate_data, 1.0f, shift_flag, rotate_flag); /* shift and rotate */
    *fitness = 0.0;    
    for (int i = 0; i < dim; i++)
        *fitness += powf(10.0f, 6.0f * i / (dim - 1)) * z[i] * z[i];
}



#endif 