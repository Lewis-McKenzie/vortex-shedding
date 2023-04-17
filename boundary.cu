#include <stdio.h>
#include <stdlib.h>

#include "data.cuh"
#include "boundary.cuh"


// loop between 1 and imax+1
#define init_outer_loop(i, limit) {int i_block_start = blockIdx.x * (imax / gridDim.x);i = i_block_start + threadIdx.x * (imax / gridDim.x) / blockDim.x + 1;limit = max(i+1, i_block_start + (threadIdx.x + 1)  * (imax / gridDim.x) / blockDim.x+1);}

// loop between 1 and jmax+1
#define init_inner_loop(j, limit) {if (blockDim.x > imax / gridDim.x) {int threads = blockDim.x / (imax / gridDim.x);int iters = jmax / threads;j = (threadIdx.x % threads) * iters + 1;limit = ((threadIdx.x % threads) + 1) * iters + 1;}else {j = 1;limit = jmax+1;}}

// changes loop to 0 -> imax+2
#define adjust_outer_loop(i, limit) {if (i == 1){i--;}if (i_end == imax+1) {i_end++;}}

// changes loop to 0 -> jmax+2
#define adjust_inner_loop(j, limit) {if (j == 1){j--;}if (j_end == jmax+1) {j_end++;}}

#define debug_cuda(i, limit) printf("thread: %d out of %d on block %d. start: %d end: %d\n", threadIdx.x, blockDim.x, blockIdx.x, i, limit);

/**
 * @brief Given the boundary conditions defined by the flag matrix, update
 * the u and v velocities. Also enforce the boundary conditions at the
 * edges of the matrix.
 */
__global__ void apply_boundary_conditions(double **u, double **v, char **flag, int imax, int jmax) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int i_start, i_end, j_start, j_end;
    init_outer_loop(i_start, i_end);


    if (i_start == 1) {
        int j, j_end;
        init_inner_loop(j, j_end)
        adjust_inner_loop(j, j_end)
        for (;j < j_end && j < jmax+2; j++) {
            /* Fluid freely flows in from the west */
            u[0][j] = u[1][j];
            v[0][j] = v[1][j];
        }
    } else if (i_end == imax+1) {
        int j, j_end;
        init_inner_loop(j, j_end)
        adjust_inner_loop(j, j_end)
        for (;j < j_end && j < jmax+2; j++) {
            /* Fluid freely flows out to the east */
            u[imax][j] = u[imax-1][j];
            v[imax+1][j] = v[imax][j];
        }
    }

    __syncthreads();
    adjust_outer_loop(i_start, i_end)
    for (int i = i_start; i < i_end && i < imax+2; i++) {
        /* The vertical velocity approaches 0 at the north and south
        * boundaries, but fluid flows freely in the horizontal direction */
        v[i][jmax] = 0.0;
        u[i][jmax+1] = u[i][jmax];

        v[i][0] = 0.0;
        u[i][0] = u[i][1];
    }

    /* Apply no-slip boundary conditions to cells that are adjacent to
     * internal obstacle cells. This forces the u and v velocity to
     * tend towards zero in these cells.
     */

    init_outer_loop(i_start, i_end);
    for (int i = i_start; i < i_end && i < imax+1; i++) {
        int j, j_end;
        init_inner_loop(j, j_end);
        for (;j >= 1 && j < j_end && j < jmax+1; j++) {
            if (flag[i][j] & B_NSEW) {
                switch (flag[i][j]) {
                    case B_N: 
                        v[i][j]   = 0.0;
                        u[i][j]   = -u[i][j+1];
                        u[i-1][j] = -u[i-1][j+1];
                        break;
                    case B_E: 
                        u[i][j]   = 0.0;
                        v[i][j]   = -v[i+1][j];
                        v[i][j-1] = -v[i+1][j-1];
                        break;
                    case B_S:
                        v[i][j-1] = 0.0;
                        u[i][j]   = -u[i][j-1];
                        u[i-1][j] = -u[i-1][j-1];
                        break;
                    case B_W: 
                        u[i-1][j] = 0.0;
                        v[i][j]   = -v[i-1][j];
                        v[i][j-1] = -v[i-1][j-1];
                        break;
                    case B_NE:
                        v[i][j]   = 0.0;
                        u[i][j]   = 0.0;
                        v[i][j-1] = -v[i+1][j-1];
                        u[i-1][j] = -u[i-1][j+1];
                        break;
                    case B_SE:
                        v[i][j-1] = 0.0;
                        u[i][j]   = 0.0;
                        v[i][j]   = -v[i+1][j];
                        u[i-1][j] = -u[i-1][j-1];
                        break;
                    case B_SW:
                        v[i][j-1] = 0.0;
                        u[i-1][j] = 0.0;
                        v[i][j]   = -v[i-1][j];
                        u[i][j]   = -u[i][j-1];
                        break;
                    case B_NW:
                        v[i][j]   = 0.0;
                        u[i-1][j] = 0.0;
                        v[i][j-1] = -v[i-1][j-1];
                        u[i][j]   = -u[i][j+1];
                        break;
                }
            }
        }
    }

    /* Finally, fix the horizontal velocity at the  western edge to have
     * a continual flow of fluid into the simulation.
     */
    __syncthreads();
    if (tid == 0) {
        v[0][0] = 2 * vi-v[1][0];
    }
    __syncthreads();
    if (i_start == 1) {
        int j, j_end;
        init_inner_loop(j, j_end);
        for (;j >= 1 && j < j_end && j < jmax+1; j++) {
            u[0][j] = ui;
            v[0][j] = 2 * vi - v[1][j];
        }
    }
}