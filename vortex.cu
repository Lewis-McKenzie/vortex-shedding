#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <time.h>

#include "data.cuh"
#include "vtk.cuh"
#include "setup.cuh"
#include "boundary.cuh"
#include "args.cuh"

struct timespec timer;
double get_time() {
	clock_gettime(CLOCK_MONOTONIC, &timer); 
	return (double) (timer.tv_sec + timer.tv_nsec / 1000000000.0);
}

#define time(func, timer) if(print_time){timer = get_time();func;timer = get_time() - timer;}else{func;}

#define print_timer(name, timer) if(print_time)printf("%s: %lf\n", name, timer);

#define init_outer_loop(i, limit, addon) i = threadIdx.x * (imax+2) / blockDim.x;limit = (threadIdx.x+1) * (imax+2) / blockDim.x;if (i == 0) {i = 1;} else if (limit > imax+addon) {i_end = imax+addon;}

#define debug_cuda(i, limit) printf("thread: %d out of %d on block %d. start: %d end: %d\n", threadIdx.x, blockDim.x, blockIdx.x, i, limit);

__device__ double reduce_sum(double value, double *reduction_buffer) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    reduction_buffer[tid] = value;
    __syncthreads();

    if (tid == 0) {
        double sum = 0;
        for (int i = 0; i < gridDim.x * blockDim.x; i++) {
            sum += reduction_buffer[i];
        }
        for (int i = 0; i < gridDim.x * blockDim.x; i++) {
            reduction_buffer[i] = sum;
        }
    }
    __syncthreads();
    return reduction_buffer[tid];
}



/**
 * @brief Computation of tentative velocity field (f, g)
 * 
 */
__device__ void compute_tentative_velocity(double** u, double **v, char **flag, double **f, double **g, int imax, int jmax, double y, double delx, double dely, double del_t) {
    int i, i_end;
    init_outer_loop(i, i_end, 0);
    //debug_cuda(i, i_end);
    for (; i < i_end; i++) {
        for (int j = 1; j < jmax+1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                double du2dx = ((u[i][j] + u[i+1][j]) * (u[i][j] + u[i+1][j]) +
                                y * fabs(u[i][j] + u[i+1][j]) * (u[i][j] - u[i+1][j]) -
                                (u[i-1][j] + u[i][j]) * (u[i-1][j] + u[i][j]) -
                                y * fabs(u[i-1][j] + u[i][j]) * (u[i-1][j]-u[i][j]))
                                / (4.0 * delx);
                double duvdy = ((v[i][j] + v[i+1][j]) * (u[i][j] + u[i][j+1]) +
                                y * fabs(v[i][j] + v[i+1][j]) * (u[i][j] - u[i][j+1]) -
                                (v[i][j-1] + v[i+1][j-1]) * (u[i][j-1] + u[i][j]) -
                                y * fabs(v[i][j-1] + v[i+1][j-1]) * (u[i][j-1] - u[i][j]))
                                / (4.0 * dely);
                double laplu = (u[i+1][j] - 2.0 * u[i][j] + u[i-1][j]) / delx / delx +
                                (u[i][j+1] - 2.0 * u[i][j] + u[i][j-1]) / dely / dely;
   
                f[i][j] = u[i][j] + del_t * (laplu / Re - du2dx - duvdy);
            } else {
                f[i][j] = u[i][j];
            }
        }
    }

    init_outer_loop(i, i_end, 1);
    for (; i < i_end; i++) {
        for (int j = 1; j < jmax; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                double duvdx = ((u[i][j] + u[i][j+1]) * (v[i][j] + v[i+1][j]) +
                                y * fabs(u[i][j] + u[i][j+1]) * (v[i][j] - v[i+1][j]) -
                                (u[i-1][j] + u[i-1][j+1]) * (v[i-1][j] + v[i][j]) -
                                y * fabs(u[i-1][j] + u[i-1][j+1]) * (v[i-1][j]-v[i][j]))
                                / (4.0 * delx);
                double dv2dy = ((v[i][j] + v[i][j+1]) * (v[i][j] + v[i][j+1]) +
                                y * fabs(v[i][j] + v[i][j+1]) * (v[i][j] - v[i][j+1]) -
                                (v[i][j-1] + v[i][j]) * (v[i][j-1] + v[i][j]) -
                                y * fabs(v[i][j-1] + v[i][j]) * (v[i][j-1] - v[i][j]))
                                / (4.0 * dely);
                double laplv = (v[i+1][j] - 2.0 * v[i][j] + v[i-1][j]) / delx / delx +
                                (v[i][j+1] - 2.0 * v[i][j] + v[i][j-1]) / dely / dely;

                g[i][j] = v[i][j] + del_t * (laplv / Re - duvdx - dv2dy);
            } else {
                g[i][j] = v[i][j];
            }
        }
    }

    /* f & g at external boundaries */
    if (threadIdx.x == 0) {
        for (int j = 1; j < jmax+1; j++) {
            f[0][j]    = u[0][j];
        }
    } else if (threadIdx.x == blockDim.x - 1) {
        for (int j = 1; j < jmax+1; j++) {
            f[imax][j] = u[imax][j];
        }
    }
    init_outer_loop(i, i_end, 1);
    for (i; i < i_end; i++) {
        g[i][0]    = v[i][0];
        g[i][jmax] = v[i][jmax];
    }
}


/**
 * @brief Calculate the right hand side of the pressure equation 
 * 
 */
__device__ void compute_rhs(char **flag, double **f, double **g, double **rhs, int imax, int jmax, double delx, double dely, double del_t) {
    int i, i_end;
    init_outer_loop(i, i_end, 1);
    for (; i < i_end; i++) {
        for (int j = 1;j < jmax+1; j++) {
            if (flag[i][j] & C_F) {
                /* only for fluid and non-surface cells */
                rhs[i][j] = ((f[i][j] - f[i-1][j]) / delx + 
                             (g[i][j] - g[i][j-1]) / dely)
                            / del_t;
            }
        }
    }
}


/**int block_dim = 16;
int grid_dim = 1;
 * @brief Red/Black SOR to solve the poisson equation.
 * 
 * @return Calculated residual of the computation
 * 
 */
__device__ double poisson(double **u, double **v, double **p, char **flag, double **rhs, int imax, int jmax, int fluid_cells, double omega, double beta_2, double rdx2, double rdy2, double *reduction_buffer) {

    double p0 = 0.0;
    /* Calculate sum of squares */
    int i, i_end;
    init_outer_loop(i, i_end, 1);
    for (; i < i_end; i++) {
        for (int j = 1; j < jmax+1; j++) {
            if (flag[i][j] & C_F) { p0 += p[i][j] * p[i][j]; }
        }
    }
    p0 = reduce_sum(p0, reduction_buffer);
   
    p0 = sqrt(p0 / fluid_cells); 
    if (p0 < 0.0001) { p0 = 1.0; }

    /* Red/Black SOR-iteration */
    int iter;
    double res = 0.0;
    for (iter = 0; iter < itermax; iter++) {

        for (int rb = 0; rb < 2; rb++) {

            init_outer_loop(i, i_end, 1);
            for (; i < i_end; i++) {
                for (int j = 1; j < jmax+1; j++) {
                    if ((i + j) % 2 != rb) { continue; }
                    if (flag[i][j] == (C_F | B_NSEW)) {
                        /* five point star for interior fluid cells */
                        p[i][j] = (1.0 - omega) * p[i][j] - 
                              beta_2 * ((p[i+1][j] + p[i-1][j] ) * rdx2
                                         + (p[i][j+1] + p[i][j-1]) * rdy2
                                         - rhs[i][j]);
                    } else if (flag[i][j] & C_F) { 
                        /* modified star near boundary */

                        double eps_E = ((flag[i+1][j] & C_F) ? 1.0 : 0.0);
                        double eps_W = ((flag[i-1][j] & C_F) ? 1.0 : 0.0);
                        double eps_N = ((flag[i][j+1] & C_F) ? 1.0 : 0.0);
                        double eps_S = ((flag[i][j-1] & C_F) ? 1.0 : 0.0);

                        double beta_mod = -omega / ((eps_E + eps_W) * rdx2 + (eps_N + eps_S) * rdy2);
                        p[i][j] = (1.0 - omega) * p[i][j] -
                            beta_mod * ((eps_E * p[i+1][j] + eps_W * p[i-1][j]) * rdx2
                                         + (eps_N * p[i][j+1] + eps_S * p[i][j-1]) * rdy2
                                         - rhs[i][j]);
                    }
                }
            }
        }
        
        /* computation of residual */
        init_outer_loop(i, i_end, 1);
        for (; i < i_end; i++) {
            for (int j = 1; j < jmax+1; j++) {
                if (flag[i][j] & C_F) {
                    double eps_E = ((flag[i+1][j] & C_F) ? 1.0 : 0.0);
                    double eps_W = ((flag[i-1][j] & C_F) ? 1.0 : 0.0);
                    double eps_N = ((flag[i][j+1] & C_F) ? 1.0 : 0.0);
                    double eps_S = ((flag[i][j-1] & C_F) ? 1.0 : 0.0);

                    /* only fluid cells */
                    double add = (eps_E * (p[i+1][j] - p[i][j]) - 
                        eps_W * (p[i][j] - p[i-1][j])) * rdx2  +
                        (eps_N * (p[i][j+1] - p[i][j]) -
                         eps_S * (p[i][j] - p[i][j-1])) * rdy2  -  rhs[i][j];
                    res += add * add;
                }
            }
        }
        res = reduce_sum(res, reduction_buffer);
        res = sqrt(res / fluid_cells) / p0;
        
        /* convergence? */
        if (res < eps) break;
    }

    return res;
}


/**
 * @brief Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
__device__ void update_velocity(double **u, double **v, double **p, char ** flag, double **f, double **g, int imax, int jmax, double delx, double dely, double del_t) {
    int i, i_end;
    init_outer_loop(i, i_end, -2);
    for (; i < i_end; i++) {
        for (int j = 1; j < jmax-1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                u[i][j] = f[i][j] - (p[i+1][j] - p[i][j]) * del_t / delx;
            }
        }
    }
    
    init_outer_loop(i, i_end, -1);
    for (; i < i_end; i++) {
        for (int j = 1; j < jmax-2; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                v[i][j] = g[i][j] - (p[i][j+1] - p[i][j]) * del_t / dely;
            }
        }
    }
}


/**
 * @brief Set the timestep size so that we satisfy the Courant-Friedrichs-Lewy
 * conditions. Otherwise the simulation becomes unstable.
 */
void set_timestep_interval() {
    /* del_t satisfying CFL conditions */
    if (tau >= 1.0e-10) { /* else no time stepsize control */
        double umax = 1.0e-10;
        double vmax = 1.0e-10; 
        
        for (int i = 0; i < imax+2; i++) {
            for (int j = 1; j < jmax+2; j++) {
                umax = fmax(fabs(u[i][j]), umax);
            }
        }

        for (int i = 1; i < imax+2; i++) {
            for (int j = 0; j < jmax+2; j++) {
                vmax = fmax(fabs(v[i][j]), vmax);
            }
        }

        double deltu = delx / umax;
        double deltv = dely / vmax; 
        double deltRe = 1.0 / (1.0 / (delx * delx) + 1 / (dely * dely)) * Re / 2.0;

        if (deltu < deltv) {
            del_t = fmin(deltu, deltRe);
        } else {
            del_t = fmin(deltv, deltRe);
        }
        del_t = tau * del_t; /* multiply by safety factor */
    }
}


__global__ void main_loop(double **u, double **v, double **p, char **flag, double **f, double **g, double **rhs, int imax, int jmax, double ui, double vi, double delx, double dely, double del_t, int fluid_cells, double omega, double beta_2, double rdx2, double rdy2, double t_end, int fixed_dt, double y, int output_freq, int no_output, int enable_checkpoints, double *reduction_buffer) {
    double res, t;

	apply_boundary_conditions(u, v, flag, imax, jmax, ui, vi);

    /* Main loop */
    int iters = 0;
    for (t = 0.0; t < t_end; t += del_t, iters++) {
        if (!fixed_dt) {
            //set_timestep_interval();
        }

        compute_tentative_velocity(u, v, flag, f, g, imax, jmax, y, delx, dely, del_t);

        compute_rhs(flag, f, g, rhs, imax, jmax, delx, dely, del_t);

        res = poisson(u, v, p, flag, rhs, imax, jmax, fluid_cells, omega, beta_2, rdx2, rdy2, reduction_buffer);

        update_velocity(u, v, p, flag, f, g, imax, jmax, delx, dely, del_t);

        apply_boundary_conditions(u, v, flag, imax, jmax, ui, vi);

        if ((iters % output_freq == 0) && blockIdx.x == 0 && threadIdx.x == 0) {
            printf("Step %8d, Time: %14.8e (del_t: %14.8e), Residual: %14.8e\n", iters, t+del_t, del_t, res);

            if ((!no_output) && (enable_checkpoints)) {
                //write_checkpoint(iters, t+del_t);
            }
        }
    } /* End of main loop */

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Step %8d, Time: %14.8e, Residual: %14.8e\n", iters, t, res);
        printf("Simulation complete.\n");
    }
}


/**
 * @brief The main routine that sets up the problem and executes the solving routines routines
 * 
 * @param argc The number of arguments passed to the program
 * @param argv An array of the arguments passed to the program
 * @return int The return value of the application
 */
int main(int argc, char *argv[]) {
    double setup_time;

    setup_time = get_time();
    set_defaults();
    parse_args(argc, argv);
    setup();

    if (verbose) print_opts();

    allocate_arrays();
    problem_set_up();
    setup_time = get_time() - setup_time;
    print_timer("Setup", setup_time);

    main_loop<<<grid_dim, block_dim>>>(u, v, p, flag, f, g, rhs, imax, jmax, ui, vi, delx, dely, del_t, fluid_cells, omega, beta_2, rdx2, rdy2, t_end, fixed_dt, y, output_freq, no_output, enable_checkpoints, reduction_buffer);
    cudaDeviceSynchronize();

    if (!no_output) {
        write_result(2000, 5);
    }

    free_arrays();

    return 0;
}
