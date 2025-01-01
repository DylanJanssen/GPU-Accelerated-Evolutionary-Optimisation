#ifndef SELECTION_CUH 
#define SELECTION_CUH 

#include <cooperative_groups.h> 
#include "cu_random.cuh"

namespace cg = cooperative_groups; 

namespace selection_warp 
{

// this tournament selection needs to split the population into num_migrants sections and each
// section will have a tournament performed in it
template <int tile_sz>
__device__ int tournament_selection(
    const cg::thread_block_tile<tile_sz> &g,
    cu_random<tile_sz> &rnd,
    float *__restrict__ population,
    float *__restrict__ fitness,
    const int dim,
    const int popsize,
    const int tournament_size,
    const int num_migrants,
    bool maximisation)
{
    int tid = cg::this_thread_block().thread_rank();
    int island_idx = g.meta_group_rank();
    int chunk_size = popsize / num_migrants;
    int parent = rnd.random_int(0, chunk_size) + island_idx * chunk_size;
    int temp_ind;

    // this forces thread 0 of the TG to perform the tournament selection
    if (g.num_threads() < tournament_size)
    {
        if (g.thread_rank() == 0)
        {
            for (int i = 0; i < tournament_size; i++)
            {
                temp_ind = rnd.random_int(0, chunk_size) + island_idx * chunk_size;
                if (maximisation && fitness[parent] < fitness[temp_ind])
                    parent = temp_ind;
                else if (!maximisation && fitness[parent] > fitness[temp_ind])
                    parent = temp_ind;
            }
        }
    }

    // otherwise use the TG accelerated tournament selection
    else //if (g.size() < tournament_size)
    {
        for (int offset = tournament_size / 2; offset > 0; offset /= 2)
        {
            temp_ind = g.shfl_down(parent, offset); // __shfl_down_sync(tournament_mask, parent, offset, tournament_size);

            if (maximisation && fitness[parent] < fitness[temp_ind])
                parent = temp_ind;
            if (!maximisation && fitness[parent] > fitness[temp_ind])
                parent = temp_ind;
        }
    }
    // broadcast the parent to the whole TG
    parent = g.shfl(parent, 0);
    return parent;
}

}
#endif 
