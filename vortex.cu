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

#define time(func, timer) if(print_time && threadIdx.x == 0){timer = get_time();func;timer = get_time() - timer;}else{func;}

#define print_timer(name, timer) if(print_time)printf("%s: %lf\n", name, timer);

#define init_outer_loop(i, limit) {int iters_per_block = (imax+2) / gridDim.x;int i_block_start = blockIdx.x * iters_per_block;i = i_block_start + threadIdx.x * iters_per_block / blockDim.x;if (blockDim.x > iters_per_block) {limit = i+1;} else {limit = i_block_start + (threadIdx.x + 1)  * iters_per_block / blockDim.x;i = max(i, 1);}}

#define init_inner_loop(j, limit) {if (blockDim.x > (imax+2) / gridDim.x) {int threads = blockDim.x / ((imax+2) / gridDim.x);int iters = (jmax+2) / threads;j = (threadIdx.x % threads) * iters;limit = ((threadIdx.x % threads) + 1) * iters;}else{j = 0;limit = jmax+2;}}


#define debug_cuda(i, limit) printf("thread: %d out of %d on block %d. start: %d end: %d\n", threadIdx.x, blockDim.x, blockIdx.x, i, limit);

__global__ void block_reduce_sum_buffer(double *reduction_buffer) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    
    for (int s=blockDim.x/2; s>0; s>>=1) {
        if (threadIdx.x < s)
            reduction_buffer[tid] += reduction_buffer[tid + s];

        __syncthreads();
    }
}

__global__ void grid_reduce_sum_buffer(double *reduction_buffer, int grid_dim, int block_dim) {
    for (int i = block_dim; i < grid_dim*block_dim; i+=block_dim) {
        reduction_buffer[0] += reduction_buffer[i];
    }
}

double reduce(double *reduction_buffer) {
    block_reduce_sum_buffer<<<grid_dim, block_dim>>>(reduction_buffer);
    grid_reduce_sum_buffer<<<1, 1>>>(reduction_buffer, grid_dim, block_dim);
    cudaDeviceSynchronize();
    return reduction_buffer[0];
}


/**
 * @brief Computation of tentative velocity field (f, g)
 * 
 */
__global__ void compute_tentative_velocity(double** u, double **v, char **flag, double **f, double **g, int imax, int jmax, double del_t, double Re, double delx, double dely) {

    int i, i_end;
    init_outer_loop(i, i_end);
    for (; i >= 1 && i < i_end && i < imax; i++) {
        int j, j_end;
        init_inner_loop(j, j_end);
        j = max(j, 1);
        for (;j >= 1 && j < j_end && j < jmax+1; j++) {
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

    init_outer_loop(i, i_end);
    for (;i >= 1 && i < i_end && i < imax+1; i++) {
        int j, j_end;
        init_inner_loop(j, j_end);
        j = max(j, 1);
        for (;j >= 1 && j < j_end && j < jmax; j++) {
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

    init_outer_loop(i, i_end);
    /* f & g at external boundaries */
    if (i == 0) {
        int j, j_end;
        init_inner_loop(j, j_end);
        j = max(j, 1);
        for (;j >= 1 &&j < j_end && j < jmax+1; j++) {
            f[0][j]    = u[0][j];
        }
    } else if (i_end == imax+2) {
        int j, j_end;
        init_inner_loop(j, j_end);
        j = max(j, 1);
        for (;j >= 1 && j < j_end && j < jmax+1; j++) {
            f[imax][j] = u[imax][j];
        }
    }
    init_outer_loop(i, i_end);
    for (; i >= 1 && i < i_end && i < imax+1; i++) {
        g[i][0]    = v[i][0];
        g[i][jmax] = v[i][jmax];
    }
}


/**
 * @brief Calculate the right hand side of the pressure equation 
 * 
 */
__global__ void compute_rhs(char **flag, double **f, double **g, double **rhs, int imax, int jmax, double del_t, double delx, double dely) {
    int i, i_end;
    init_outer_loop(i, i_end);
    for (;i >=1 && i < i_end && i < imax+1; i++) {
        int j, j_end;
        init_inner_loop(j, j_end);
        j = max(j, 1);
        for (;j >= 1 && j < j_end && j < jmax+1; j++) {
            if (flag[i][j] & C_F) {
                /* only for fluid and non-surface cells */
                rhs[i][j] = ((f[i][j] - f[i-1][j]) / delx + 
                             (g[i][j] - g[i][j-1]) / dely)
                            / del_t;
            }
        }
    }
}


__global__ void init_p0(double **p, char **flag, int imax, int jmax, double *reduction_buffer) {
    double p0 = 0.0;
    /* Calculate sum of squares */
    int i, i_end;
    init_outer_loop(i, i_end);
    for (;i >= 1 && i < i_end && i < imax+1; i++) {
        int j, j_end;
        init_inner_loop(j, j_end);
        j = max(j, 1);
        for (;j >= 1 && j < j_end && j < jmax+1; j++) {
            if (flag[i][j] & C_F) { p0 += p[i][j] * p[i][j]; }
        }
    }
    reduction_buffer[blockIdx.x * blockDim.x + threadIdx.x] = p0;
}


__global__ void update_p(double **p, char **flag, double **rhs, int imax, int jmax) {
    int i, i_end;
    for (int rb = 0; rb < 2; rb++) {

        init_outer_loop(i, i_end);
        for (;i >= 1 && i < i_end && i < imax+1; i++) {
            int j, j_end;
            init_inner_loop(j, j_end);
            if (i > 510) {
                printf("block: %d thread: %d i start %d end %d j start %d end %d\n", blockIdx.x, threadIdx.x, i, i_end, j, j_end);
            }
            return;
            j = max(j, 1);
            for (;j >= 1 && j < j_end && j < jmax+1; j++) {
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

}

__global__ void update_res(double **p, char **flag, double **rhs, int imax, int jmax, double res, double *reduction_buffer) {
    /* computation of residual */
    int i, i_end;
    init_outer_loop(i, i_end);
    for (;i >= 1 && i < i_end && i < imax+1; i++) {
        int j, j_end;
        init_inner_loop(j, j_end);
        j = max(j, 1);
        for (;j >= 1 && j < j_end && j < jmax+1; j++) {
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
    reduction_buffer[blockIdx.x * blockDim.x + threadIdx.x] = res;
}

/**
 * @brief Red/Black SOR to solve the poisson equation.
 * 
 * @return Calculated residual of the computation
 * 
 */
double poisson() {

    init_p0<<<grid_dim, block_dim>>>(p, flag, imax, jmax, reduction_buffer);
    double p0 = reduce(reduction_buffer);
   
    p0 = sqrt(p0 / fluid_cells); 
    if (p0 < 0.0001) { p0 = 1.0; }

    /* Red/Black SOR-iteration */
    int iter;
    double res = 0.0;
    for (iter = 0; iter < itermax; iter++) {

        update_p<<<grid_dim, block_dim>>>(p, flag, rhs, imax, jmax);
        return 0.0;
        update_res<<<grid_dim, block_dim>>>(p, flag, rhs, imax, jmax, res, reduction_buffer);
        res = reduce(reduction_buffer);

        res = sqrt(res / fluid_cells) / p0;
        //printf("%lf\n", res);

        /* convergence? */
        if (res < eps) break;
    }

    return res;
}


/**
 * @brief Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
__global__ void update_velocity(double **u, double **v, double **p, char ** flag, double **f, double **g, int imax, int jmax, double del_t, double delx, double dely) {
    int i, i_end;
    init_outer_loop(i, i_end);
    for (;i >= 1 && i < i_end && i < imax-2; i++) {
        int j, j_end;
        init_inner_loop(j, j_end);
        j = max(j, 1);
        for (;j >= 1 && j < j_end && j < jmax-1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                u[i][j] = f[i][j] - (p[i+1][j] - p[i][j]) * del_t / delx;
            }
        }
    }
    
    init_outer_loop(i, i_end);
    for (;i >= 1 && i < i_end && i < imax-1; i++) {
        int j, j_end;
        init_inner_loop(j, j_end);
        j = max(j, 1);
        for (;j >= 1 && j < j_end && j < jmax-2; j++) {
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


void main_loop() {
    double res, t, ten_t, rhs_t, pois_t, vel_t, bound_t;

    apply_boundary_conditions<<<grid_dim, block_dim>>>(u, v, flag, imax, jmax);

    /* Main loop */
    int iters = 0;
    for (t = 0.0; t < t_end; t += del_t, iters++) {
        if (!fixed_dt) {
            set_timestep_interval();
        }

        (compute_tentative_velocity<<<grid_dim, block_dim>>>(u, v, flag, f, g, imax, jmax, del_t, Re, delx, dely), ten_t);

        (compute_rhs<<<grid_dim, block_dim>>>(flag, f, g, rhs, imax, jmax, del_t, delx, dely), rhs_t);

        (res = poisson(), pois_t);

        (update_velocity<<<grid_dim, block_dim>>>(u, v, p, flag, f, g, imax, jmax, del_t, delx, dely), vel_t);

        (apply_boundary_conditions<<<grid_dim, block_dim>>>(u, v, flag, imax, jmax), bound_t);

        if ((iters % output_freq == 0)) {
            cudaDeviceSynchronize();
            printf("Step %8d, Time: %14.8e (del_t: %14.8e), Residual: %14.8e\n", iters, t+del_t, del_t, res);

            if ((!no_output) && (enable_checkpoints)) {
                write_checkpoint(iters, t+del_t);
            }
        }
    } /* End of main loop */

    printf("Step %8d, Time: %14.8e, Residual: %14.8e\n", iters, t, res);
    printf("Simulation complete.\n");

    if (!no_output) {
        write_result(iters, t);
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
    cudaDeviceSynchronize();

    if (verbose) print_opts();

    allocate_arrays();
    problem_set_up();
    setup_time = get_time() - setup_time;
    print_timer("Setup", setup_time);

    main_loop();


    free_arrays();

    return 0;
}
