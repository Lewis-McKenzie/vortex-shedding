#include "data.h"
#include "boundary.h"

/**
 * @brief Given the boundary conditions defined by the cuda_flag matrix, update
 * the cuda_u and cuda_v velocities. Also enforce the boundary conditions at the
 * edges of the matrix.
 */
__global__ void apply_boundary_conditions(int imax, int jmax, double ui, double vi) {
    for (int j = 0; j < jmax+2; j++) {
        /* Fluid freely flows in from the west */
        cuda_u[0][j] = cuda_u[1][j];
        cuda_v[0][j] = cuda_v[1][j];

        /* Fluid freely flows out to the east */
        cuda_u[imax][j] = cuda_u[imax-1][j];
        cuda_v[imax+1][j] = cuda_v[imax][j];
    }

    for (int i = 0; i < imax+2; i++) {
        /* The vertical velocity approaches 0 at the north and south
         * boundaries, but fluid flows freely in the horizontal direction */
        cuda_v[i][jmax] = 0.0;
        cuda_u[i][jmax+1] = cuda_u[i][jmax];

        cuda_v[i][0] = 0.0;
        cuda_u[i][0] = cuda_u[i][1];
    }

    /* Apply no-slip boundary conditions to cells that are adjacent to
     * internal obstacle cells. This forces the cuda_u and cuda_v velocity to
     * tend towards zero in these cells.
     */
    for (int i = 1; i < imax+1; i++) {
        for (int j = 1; j < jmax+1; j++) {
            if (cuda_flag[i][j] & B_NSEW) {
                switch (cuda_flag[i][j]) {
                    case B_N: 
                        cuda_v[i][j]   = 0.0;
                        cuda_u[i][j]   = -cuda_u[i][j+1];
                        cuda_u[i-1][j] = -cuda_u[i-1][j+1];
                        break;
                    case B_E: 
                        cuda_u[i][j]   = 0.0;
                        cuda_v[i][j]   = -cuda_v[i+1][j];
                        cuda_v[i][j-1] = -cuda_v[i+1][j-1];
                        break;
                    case B_S:
                        cuda_v[i][j-1] = 0.0;
                        cuda_u[i][j]   = -cuda_u[i][j-1];
                        cuda_u[i-1][j] = -cuda_u[i-1][j-1];
                        break;
                    case B_W: 
                        cuda_u[i-1][j] = 0.0;
                        cuda_v[i][j]   = -cuda_v[i-1][j];
                        cuda_v[i][j-1] = -cuda_v[i-1][j-1];
                        break;
                    case B_NE:
                        cuda_v[i][j]   = 0.0;
                        cuda_u[i][j]   = 0.0;
                        cuda_v[i][j-1] = -cuda_v[i+1][j-1];
                        cuda_u[i-1][j] = -cuda_u[i-1][j+1];
                        break;
                    case B_SE:
                        cuda_v[i][j-1] = 0.0;
                        cuda_u[i][j]   = 0.0;
                        cuda_v[i][j]   = -cuda_v[i+1][j];
                        cuda_u[i-1][j] = -cuda_u[i-1][j-1];
                        break;
                    case B_SW:
                        cuda_v[i][j-1] = 0.0;
                        cuda_u[i-1][j] = 0.0;
                        cuda_v[i][j]   = -cuda_v[i-1][j];
                        cuda_u[i][j]   = -cuda_u[i][j-1];
                        break;
                    case B_NW:
                        cuda_v[i][j]   = 0.0;
                        cuda_u[i-1][j] = 0.0;
                        cuda_v[i][j-1] = -cuda_v[i-1][j-1];
                        cuda_u[i][j]   = -cuda_u[i][j+1];
                        break;
                }
            }
        }
    }

    /* Finally, fix the horizontal velocity at the  western edge to have
     * a continual flow of fluid into the simulation.
     */
    cuda_v[0][0] = 2 * vi-cuda_v[1][0];
    for (int j = 1; j < jmax+1; j++) {
        cuda_u[0][j] = ui;
        cuda_v[0][j] = 2 * vi - cuda_v[1][j];
    }
}