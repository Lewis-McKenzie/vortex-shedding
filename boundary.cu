#include "data.cuh"
#include "boundary.cuh"

#define init_outer_loop(i, limit, addon) i = threadIdx.x * (imax+2) / blockDim.x;limit = (threadIdx.x+1) * (imax+2) / blockDim.x;if (i == 0) {i = 1;} else if (limit > imax+addon) {i_end = imax+addon;}

/**
 * @brief Given the boundary conditions defined by the flag matrix, update
 * the u and v velocities. Also enforce the boundary conditions at the
 * edges of the matrix.
 */
__device__ void apply_boundary_conditions(double **u, double **v, char **flag, int imax, int jmax, double ui, double vi) {
    int i, i_end;
    init_outer_loop(i, i_end, 2);
    
    if (threadIdx.x == 0) {
        for (int j = 0; j < jmax+2; j++) {
            /* Fluid freely flows in from the west */
            u[0][j] = u[1][j];
            v[0][j] = v[1][j];
        }
        i = 0;
    } else if (threadIdx.x == blockDim.x - 1) {
        for (int j = 0; j < jmax+2; j++) {
            /* Fluid freely flows out to the east */
            u[imax][j] = u[imax-1][j];
            v[imax+1][j] = v[imax][j];
        }
    }

    for (; i < i_end; i++) {
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

    init_outer_loop(i, i_end, 1);
    for (; i < i_end; i++) {
        for (int j = 1; j < jmax+1; j++) {
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
    if (threadIdx.x == 0) {
        v[0][0] = 2 * vi-v[1][0];
        for (int j = 1; j < jmax+1; j++) {
            u[0][j] = ui;
            v[0][j] = 2 * vi - v[1][j];
        }
    }
}