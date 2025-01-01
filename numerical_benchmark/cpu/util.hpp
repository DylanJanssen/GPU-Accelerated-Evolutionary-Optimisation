#ifndef UTIL_H 
#define UTIL_H

// Utility functions for benchmarks 


void scale(float *x, float *x_scale, int dim, float scale_rate);
void shift(float *x, float *x_shift, int dim, float *shift_data);
void rotate(float *x, float *x_rotate, int dim, float *rotate_data);
void scale_shift_and_rotate(float *x, float *y, float *z, int dim, float *shift_data, float *rotate_data, float scale_rate, bool shift_flag, bool rotate_flag); /* shift and rotate */
void shuffle(float *x, float *x_shuffle, int dim, int *shuffle_data);


void scale(
    float *x, 
    float *x_scale, 
    int dim, 
    float scale_rate)
{
    for (int i = 0; i < dim; i++)
        x_scale[i] = x[i] * scale_rate; 
}

void shift(
    float *x, 
    float *x_shift, 
    int dim, 
    float *shift_data)
{
    for (int i = 0; i < dim; i++)
        x_shift[i] = x[i] - shift_data[i];
}

void rotate(
    float *x, 
    float *x_rotate, 
    int dim, 
    float *rotate_data)
{
    for (int i = 0; i < dim; i++)
    {
        x_rotate[i] = 0.0f;
        for (int j = 0; j < dim; j++)
            x_rotate[i] += x[j] * rotate_data[i * dim + j];
    }
}

void scale_shift_and_rotate(
    float *x, 
    float *y, 
    float *z, 
    int dim, 
    float *shift_data, 
    float *rotate_data, 
    float scale_rate, 
    bool shift_flag, 
    bool rotate_flag)
{
    if (shift_flag)
    {
        shift(x, y, dim, shift_data); // shift x and store in y
        if (rotate_flag)
        {
            scale(y, y, dim, scale_rate);   // scale y and store in y
            rotate(y, z, dim, rotate_data); // rotate y and store in z
        }
        else
            scale(y, z, dim, scale_rate); // scale y and store in z
    }
    else if (rotate_flag)
    {
        scale(x, y, dim, scale_rate);   // scale x and store in y
        rotate(y, z, dim, rotate_data); // rotate y and store in z
    }
    else
        scale(x, z, dim, scale_rate); // scale x and store in z
}


void shuffle(
    float *x,
    float *x_shuffle,
    const int dim,
    int *shuffle_data)
{
    for (int i = 0; i < dim; i++)
        x_shuffle[i] = x[shuffle_data[i] - 1];
}

#endif 