#ifndef SOLUTION_CUH 
#define SOLUTION_CUH 

/*
    Used for CUDA based evolutionary algorithms to store 
    whether a solution has been found and the index of the 
    individual in the overall population that found the solution. 
    Solution iteration is used for algorithms that run multiple 
    iterations in a kernel 
*/
struct solution 
{
    int solution_found; 
    int solution_iteration; 
    int solution_idx; 
    solution() : solution_found(0), solution_iteration(0), solution_idx(0){}
};

/*
    device function that checks if a solution has been found. Should only 
    be called by a single thread that represents the indiviudal. 
*/
__device__ __forceinline__
void check_solution(float *__restrict__ fitness, solution *sol, int index) 
{
    if (*fitness < 10e-8)
    {
        *fitness = 0.0f; 
        int val = atomicAdd(&sol->solution_found, 1); 
        if (val == 0) 
            sol->solution_idx = index; 
    }
}

#endif 