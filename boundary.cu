#include "data.cuh"
#include "boundary.cuh"


#define init_outer_loop(i, limit) {int iters_per_block = (imax+2) / gridDim.x;int i_block_start = blockIdx.x * iters_per_block;i = i_block_start + threadIdx.x * iters_per_block / blockDim.x;if (blockDim.x > iters_per_block) {limit = i+1;} else {limit = i_block_start + (threadIdx.x + 1)  * iters_per_block / blockDim.x;}}

#define init_inner_loop(j, limit) {if (blockDim.x > (imax+2) / gridDim.x) {int threads = blockDim.x / ((imax+2) / gridDim.x);int iters = (jmax+2) / threads;j = (threadIdx.x % threads) * iters;j_end = ((threadIdx.x % threads) + 1) * iters;}else{j = 0;limit = jmax+2;}}

/**
 * @brief Given the boundary conditions defined by the flag matrix, update
 * the u and v velocities. Also enforce the boundary conditions at the
 * edges of the matrix.
 */
__global__ void apply_boundary_conditions(double **u, double **v, char **flag, int imax, int jmax) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int i, i_end;
    init_outer_loop(i, i_end);

    if (i == 0) {
        int j, j_end;
        init_inner_loop(j, j_end)
        for (;j < j_end && j < jmax+2; j++) {
            /* Fluid freely flows in from the west */
            u[0][j] = u[1][j];
            v[0][j] = v[1][j];
        }
    } else if (i_end == imax+2) {
        int j, j_end;
        init_inner_loop(j, j_end)
        for (int j = 0; j < jmax+2; j++) {
            /* Fluid freely flows out to the east */
            u[imax][j] = u[imax-1][j];
            v[imax+1][j] = v[imax][j];
        }
    }

    for (; i < i_end && i < imax+2; i++) {
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

    init_outer_loop(i, i_end);
    if (blockDim.x < (imax+2) / gridDim.x) {
        i = max(i, 1);
    }
    for (;i >= 1 && i < i_end && i < imax+1; i++) {
        int j, j_end;
        init_inner_loop(j, j_end);
        j = max(j, 1);
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
    if (tid == 0) {
        v[0][0] = 2 * vi-v[1][0];
    }
    init_outer_loop(i, i_end);
    if (i == 0) {
        int j, j_end;
        init_inner_loop(j, j_end);
        j = max(j, 1);
        for (;j >= 1 && j < j_end && j < jmax+1; j++) {
            u[0][j] = ui;
            v[0][j] = 2 * vi - v[1][j];
        }
    }
}