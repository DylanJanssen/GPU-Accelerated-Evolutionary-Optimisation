#ifndef BLOCK_BASIC_FUNCTIONS_CUH 
#define BLOCK_BASIC_FUNCTIONS_CUH 

#include <cooperative_groups.h> 
#include "util.cuh"

// E is used for Ackley function
#define E 2.7182818284590452353602874713526625

namespace benchmarks_block 
{

namespace cg = cooperative_groups;

__device__ __forceinline__ void zakharov_function(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ y,
    float *__restrict__ z,
    float *__restrict__ fitness,
    int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    bool shift_flag,
    bool rotate_flag)
{
    scale_shift_and_rotate(g, x, y, z, dim, shift_data, rotate_data, 1.0f, shift_flag, rotate_flag);
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = g.thread_rank(); i < dim; i += g.size()) // calculation
    {
        sum1 += z[i] * z[i];
        sum2 += 0.5f * (i + 1) * z[i];
    }
    // parallel reduction
    sum1 = reduce_sum(g, y, sum1, dim);
    sum2 = reduce_sum(g, y, sum2, dim);

    if (g.thread_rank() == 0)
        *fitness = sum1 + sum2 * sum2 + powf(sum2, 4);
    g.sync(); 
}

// GPU benchmark functions
__device__ __forceinline__ void rosenbrock_function(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ y,
    float *__restrict__ z,
    float *__restrict__ fitness,
    int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    bool shift_flag,
    bool rotate_flag)
{
    scale_shift_and_rotate(g, x, y, z, dim, shift_data, rotate_data, 0.02048f, shift_flag, rotate_flag);
    // shift to origin
    for (int i = g.thread_rank(); i < dim; i += g.size())
        z[i] += 1.0f;
    g.sync();
    // rosenbrock function
    float temp1, temp2, partial_solution = 0.0f;
    for (int i = g.thread_rank(); i < dim - 1; i += g.size())
    {
        temp1 = z[i] * z[i] - z[i + 1];
        temp2 = z[i] - 1.0f;
        partial_solution += 100.0f * temp1 * temp1 + temp2 * temp2;
    }
    // parallel reduction of partial solutions, uses y as temporary storage, thread 0 contains final sum
    partial_solution = reduce_sum(g, y, partial_solution, dim-1);
    if (g.thread_rank() == 0)
        *fitness = partial_solution;
    g.sync(); 
}

__device__ __forceinline__ void rastrigin_function(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ y,
    float *__restrict__ z,
    float *__restrict__ fitness,
    int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    bool shift_flag,
    bool rotate_flag)
{
    scale_shift_and_rotate(g, x, y, z, dim, shift_data, rotate_data, 0.0512f, shift_flag, rotate_flag);
    float s = 0.0f;
    for (int i = g.thread_rank(); i < dim; i += g.size()) // calculation
        s += (z[i] * z[i] - 10.0f * cospif(2.0f * z[i]) + 10.0f);
    s = reduce_sum(g, y, s, dim); // parallel reduction
    if (g.thread_rank() == 0)     // thread 0 assigns final value
        *fitness = s;
    g.sync(); 
}

__device__ __forceinline__ void schwefel_function(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ y,
    float *__restrict__ z,
    float *__restrict__ fitness,
    int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    bool shift_flag,
    bool rotate_flag)
{
    scale_shift_and_rotate(g, x, y, z, dim, shift_data, rotate_data, 10.0f, shift_flag, rotate_flag);
    float s = 0.0f, tmp;

    for (int i = g.thread_rank(); i < dim; i += g.size())
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
    s = reduce_sum(g, y, s, dim); // parallel reductin
    if (g.thread_rank() == 0)
        *fitness = s + 418.9828872724338f * dim;
    g.sync(); 
}

// hybrid functions

// hybrid 1 is comprised of bent cigar, hgbat and rastrigin
__device__ __forceinline__ void bent_cigar_function(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ y,
    float *__restrict__ z,
    float *__restrict__ fitness,
    int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    bool shift_flag,
    bool rotate_flag)
{
    scale_shift_and_rotate(g, x, y, z, dim, shift_data, rotate_data, 1.0f, shift_flag, rotate_flag);
    float s = 0.0f;
    if (g.thread_rank() == 0)
        s = z[0] * z[0]; 
    else 
        for (int i = g.thread_rank(); i < dim; i += g.size())
            s += 1000000.0f * z[i] * z[i];
    s = reduce_sum(g, y, s, dim); // parallel reduction
    if (g.thread_rank() == 0)
        *fitness = s;
    g.sync(); 
}

__device__ __forceinline__ void hgbat_function(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ y,
    float *__restrict__ z,
    float *__restrict__ fitness,
    int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    bool shift_flag,
    bool rotate_flag)
{
    scale_shift_and_rotate(g, x, y, z, dim, shift_data, rotate_data, 0.05f, shift_flag, rotate_flag);
    float s = 0.0f, alpha=0.25f, r2 = 0.0f;

    for (int i = g.thread_rank(); i < dim; i += g.size()) // calculation
    {
        z[i] = z[i] - 1.0f; // shift to origin
        r2 += z[i] * z[i];
        s += z[i];
    }
    r2 = reduce_sum(g, y, r2, dim);
    s = reduce_sum(g, y, s, dim);
    if (g.thread_rank() == 0)
        *fitness = powf(fabsf(r2 * r2 - s * s), 2 * alpha) + (0.5f * r2 + s) / dim + 0.5f;
    g.sync(); 
}


// hybrid 2 is comprised of escaffer6, hgbat, rosenbrock, schwefel
__device__ __forceinline__ void expanded_schaffer_F6_function(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ y,
    float *__restrict__ z,
    float *__restrict__ fitness,
    int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    bool shift_flag,
    bool rotate_flag)
{
    scale_shift_and_rotate(g, x, y, z, dim, shift_data, rotate_data, 1.0f, shift_flag, rotate_flag);
    float s = 0.0f, temp1, temp2, a;
    for (int i = g.thread_rank(); i < dim; i += g.size())
    {
        a = z[i] * z[i] + z[(i + 1) % dim] * z[(i + 1) % dim];
        temp1 = sinf(sqrtf(a));
        temp1 *= temp1;
        temp2 = 1.0f + 0.001f * a;
        s += 0.5f + (temp1 - 0.5f) / (temp2 * temp2);
    }
    s = reduce_sum(g, y, s, dim); // parallel reduction
    if (g.thread_rank() == 0)
        *fitness = s;
    g.sync(); 
}


// hybrid function 3 is comprised of 5 functions
// katsuura, happycat, grie rosen, schwefel, ackley
__device__ __forceinline__ void katsuura_function(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ y,
    float *__restrict__ z,
    float *__restrict__ fitness,
    int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    bool shift_flag,
    bool rotate_flag)
{
    scale_shift_and_rotate(g, x, y, z, dim, shift_data, rotate_data, 0.05f, shift_flag, rotate_flag);
    float s = 1.0f;
    float temp1, temp2, temp3, temp4;
    temp4 = powf(1.0f * dim, 1.2f);
    for (int i = g.thread_rank(); i < dim; i += g.size())
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
    s = reduce_product(g, y, s, dim); // parallel reduction
    if (g.thread_rank() == 0)
    {
        temp2 = 10.0f / dim / dim; 
        *fitness = s * temp2 - temp2;
    }
    g.sync(); 
}

__device__ __forceinline__ void happycat_function(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ y,
    float *__restrict__ z,
    float *__restrict__ fitness,
    int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    bool shift_flag,
    bool rotate_flag)
{
    scale_shift_and_rotate(g, x, y, z, dim, shift_data, rotate_data, 0.05f, shift_flag, rotate_flag);
    float s = 0.0f;
    float r = 0.0f;
    float alpha = 0.125f;

    for (int i = g.thread_rank(); i < dim; i += g.size())
    {
        z[i] -= 1.0f; // shift to origin
        r += z[i] * z[i];
        s += z[i];
    }
    s = reduce_sum(g, y, s, dim); // parallel reduction
    r = reduce_sum(g, y, r, dim); // parallel reduction
    if (g.thread_rank() == 0)
        *fitness = powf(fabsf(r - dim), 2 * alpha) + (0.5f * r + s) / dim + 0.5f;
    g.sync(); 
}

__device__ __forceinline__ void expanded_griewank_plus_rosenbrock_function(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ y,
    float *__restrict__ z,
    float *__restrict__ fitness,
    int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    bool shift_flag,
    bool rotate_flag)
{
    scale_shift_and_rotate(g, x, y, z, dim, shift_data, rotate_data, 0.05f, shift_flag, rotate_flag);
    float s = 0.0f;
    float temp1, temp2, temp3;
    for (int i = g.thread_rank(); i < dim; i += g.size()) // shift to origin 
        z[i] += 1.0f; 
    g.sync(); 
    for (int i = g.thread_rank(); i < dim; i += g.size())
    {
        // rosenbrock 
        temp1 = z[i] * z[i] - z[(i + 1) % dim];
        temp2 = z[i] - 1.0f;
        temp3 = 100.0f * temp1 * temp1 + temp2 * temp2;
        // end rosenbrock
        // griewank, shifts to origin
        s += (temp3 * temp3) / 4000.0f - cosf(temp3) + 1.0f;
    }
    s = reduce_sum(g, y, s, dim); // parallel reduction
    if (g.thread_rank() == 0)
        *fitness = s;
    g.sync(); 
}

__device__ __forceinline__ void ackley_function(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ y,
    float *__restrict__ z,
    float *__restrict__ fitness,
    int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    bool shift_flag,
    bool rotate_flag)
{
    scale_shift_and_rotate(g, x, y, z, dim, shift_data, rotate_data, 1.0f, shift_flag, rotate_flag);
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for (int i = g.thread_rank(); i < dim; i += g.size())
    {
        sum1 += z[i] * z[i];
        sum2 += cospif(2.0f * z[i]);
    }
    sum1 = reduce_sum(g, y, sum1, dim); // parallel reduction
    sum2 = reduce_sum(g, y, sum2, dim); // parallel reduction
    if (g.thread_rank() == 0)
    {
        sum1 = -0.2f * sqrtf(sum1 / dim);
        sum2 /= dim;
        *fitness = E - 20.0f * expf(sum1) - expf(sum2) + 20.0f;
    }
    g.sync(); 
}


// composition functions
__device__ __forceinline__ void griewank_function(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ y,
    float *__restrict__ z,
    float *__restrict__ fitness,
    int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    bool shift_flag,
    bool rotate_flag)
{
    scale_shift_and_rotate(g, x, y, z, dim, shift_data, rotate_data, 6.0f, shift_flag, rotate_flag);
    float s = 0.0f;
    float p = 1.0f; 
    for (int i = g.thread_rank(); i < dim; i += g.size())
    {
        s += z[i] * z[i]; 
        p *= cosf(z[i] / sqrtf(1.0f + i));
    }
    s = reduce_sum(g, y, s, dim); // parallel reduction
    p = reduce_product(g, y, p, dim); // parallel reduction
    if (g.thread_rank() == 0)
    {
        *fitness = 1.0f + s / 4000.0f - p;
        // printf("griewank %f %f %f\n", s, p, *fitness);
    }

    g.sync(); 
}


// composition function 2 comprised of ackley, ellipsoidal, griewank, rastrigin
__device__ __forceinline__ void high_conditional_elliptic_function(
    const cg::thread_block &g,
    float *__restrict__ x,
    float *__restrict__ y,
    float *__restrict__ z,
    float *__restrict__ fitness,
    int dim,
    float *__restrict__ shift_data,
    float *__restrict__ rotate_data,
    int *__restrict__ shuffle_data,
    bool shift_flag,
    bool rotate_flag)
{
    scale_shift_and_rotate(g, x, y, z, dim, shift_data, rotate_data, 1.0f, shift_flag, rotate_flag);
    float s = 0.0f;
    for (int i = g.thread_rank(); i < dim; i += g.size())
        s += powf(10.0f, 6.0f * i / (dim - 1)) * z[i] * z[i]; 
    s = reduce_sum(g, y, s, dim); // parallel reduction
    if (g.thread_rank() == 0)
        *fitness = s;
    g.sync(); 
}

}

#endif 