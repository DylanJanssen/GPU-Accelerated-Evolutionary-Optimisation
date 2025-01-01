#ifndef CUDA_ERR_H
#define CUDA_ERR_H

#define DEBUG 1
#ifdef DEBUG
// Macro function to wrap around CUDA calls for error checking
#define cuda_error_check(call)                                                 \
    {                                                                          \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                           \
        }                                                                      \
    }
#else
#define cuda_error_check(call) \
    {                          \
    }
#endif

#endif 