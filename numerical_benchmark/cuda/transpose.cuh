#ifndef TRANSPOSE_CUH 
#define TRANSPOSE_CUH 

#define BLOCK_DIM 32

__global__ void transpose(float *output_data, float *input_data, int width, int height)
{
    __shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];

    // read the matrix tile into shared memory in transposed order
    int x_idx = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int y_idx = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if ((x_idx < width) && (y_idx < height))
        block[threadIdx.y][threadIdx.x] = input_data[y_idx * width + x_idx];

    __syncthreads();

    // write the transposed matrix tile to global memory (output_data) in linear order
    x_idx = blockIdx.y * BLOCK_DIM + threadIdx.x;
    y_idx = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if ((x_idx < height) && (y_idx < width))
        output_data[y_idx * height + x_idx] = block[threadIdx.x][threadIdx.y];
}

#endif 