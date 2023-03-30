#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

/**
 * @brief Computation of tentative velocity field (f, g)
 * 
 */
__global__ void compute_tentative_velocity(int imax, int jmax, double delx, double dely, double del_t, double Re, double y) {
    for (int i = 1; i < imax; i++) {
        for (int j = 1; j < jmax+1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((cuda_flag[at(i, j)] & C_F) && (cuda_flag[at(i+1, j)] & C_F)) {
                double du2dx = ((cuda_u[at(i, j)] + cuda_u[at(i+1, j)]) * (cuda_u[at(i, j)] + cuda_u[at(i+1, j)]) +
                                y * fabs(cuda_u[at(i, j)] + cuda_u[at(i+1, j)]) * (cuda_u[at(i, j)] - cuda_u[at(i+1, j)]) -
                                (cuda_u[at(i-1, j)] + cuda_u[at(i, j)]) * (cuda_u[at(i-1, j)] + cuda_u[at(i, j)]) -
                                y * fabs(cuda_u[at(i-1, j)] + cuda_u[at(i, j)]) * (cuda_u[at(i-1, j)] - cuda_u[at(i, j)]))
                                / (4.0 * delx);
                double duvdy = ((cuda_v[at(i, j)] + cuda_v[at(i+1, j)]) * (cuda_u[at(i, j)] + cuda_u[at(i, j+1)]) +
                                y * fabs(cuda_v[at(i, j)] + cuda_v[at(i+1, j)]) * (cuda_u[at(i, j)] - cuda_u[at(i, j+1)]) -
                                (cuda_v[at(i, j-1)] + cuda_v[at(i+1, j-1)]) * (cuda_u[at(i, j-1)] + cuda_u[at(i, j)]) -
                                y * fabs(cuda_v[at(i, j-1)] + cuda_v[at(i+1, j-1)]) * (cuda_u[at(i, j-1)] - cuda_u[at(i, j)]))
                                / (4.0 * dely);
                double laplu = (cuda_u[at(i+1, j)] - 2.0 * cuda_u[at(i, j)] + cuda_u[at(i-1, j)]) / delx / delx +
                                (cuda_u[at(i, j+1)] - 2.0 * cuda_u[at(i, j)] + cuda_u[at(i, j-1)]) / dely / dely;
   
                cuda_f[at(i, j)] = cuda_u[at(i, j)] + del_t * (laplu / Re - du2dx - duvdy);
            } else {
                cuda_f[at(i, j)] = cuda_u[at(i, j)];
            }
        }
    }
    for (int i = 1; i < imax+1; i++) {
        for (int j = 1; j < jmax; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((cuda_flag[at(i, j)] & C_F) && (cuda_flag[at(i, j+1)] & C_F)) {
                double duvdx = ((cuda_u[at(i, j)] + cuda_u[at(i, j+1)]) * (cuda_v[at(i, j)] + cuda_v[at(i+1, j)]) +
                                y * fabs(cuda_u[at(i, j)] + cuda_u[at(i, j+1)]) * (cuda_v[at(i, j)] - cuda_v[at(i+1, j)]) -
                                (cuda_u[at(i-1, j)] + cuda_u[at(i-1, j+1)]) * (cuda_v[at(i-1, j)] + cuda_v[at(i, j)]) -
                                y * fabs(cuda_u[at(i-1, j)] + cuda_u[at(i-1, j+1)]) * (cuda_v[at(i-1, j)] - cuda_v[at(i, j)]))
                                / (4.0 * delx);
                double dv2dy = ((cuda_v[at(i, j)] + cuda_v[at(i, j+1)]) * (cuda_v[at(i, j)] + cuda_v[at(i, j+1)]) +
                                y * fabs(cuda_v[at(i, j)] + cuda_v[at(i, j+1)]) * (cuda_v[at(i, j)] - cuda_v[at(i, j+1)]) -
                                (cuda_v[at(i, j-1)] + cuda_v[at(i, j)]) * (cuda_v[at(i, j-1)] + cuda_v[at(i, j)]) -
                                y * fabs(cuda_v[at(i, j-1)] + cuda_v[at(i, j)]) * (cuda_v[at(i, j-1)] - cuda_v[at(i, j)]))
                                / (4.0 * dely);
                double laplv = (cuda_v[at(i+1, j)] - 2.0 * cuda_v[at(i, j)] + cuda_v[at(i-1, j)]) / delx / delx +
                                (cuda_v[at(i, j+1)] - 2.0 * cuda_v[at(i, j)] + cuda_v[at(i, j-1)]) / dely / dely;

                cuda_g[at(i, j)] = cuda_v[at(i, j)] + del_t * (laplv / Re - duvdx - dv2dy);
            } else {
                cuda_g[at(i, j)] = cuda_v[at(i, j)];
            }
        }
    }

    /* f & g at external boundaries */
    for (int j = 1; j < jmax+1; j++) {
        cuda_f[at(0, j)]    = cuda_u[at(0, j)];
        cuda_f[at(imax, j)] = cuda_u[at(imax, j)];
    }
    for (int i = 1; i < imax+1; i++) {
        cuda_g[at(i, 0)]    = cuda_v[at(i, 0)];
        cuda_g[at(i, jmax)] = cuda_v[at(i, jmax)];
    }
}


/**
 * @brief Calculate the right hand side of the pressure equation 
 * 
 */
__global__ void compute_rhs(int imax, int jmax, int delx, double dely, double del_t) {
    for (int i = 1; i < imax+1; i++) {
        for (int j = 1;j < jmax+1; j++) {
            if (cuda_flag[at(i, j)] & C_F) {
                /* only for fluid and non-surface cells */
                cuda_rhs[at(i, j)] = ((cuda_f[at(i, j)] - cuda_f[at(i-1, j)]) / delx + 
                             (cuda_g[at(i, j)] - cuda_g[at(i, j-1)]) / dely)
                            / del_t;
            }
        }
    }
}

/**
 * @brief Red/Black SOR to solve the poisson equation.
 * 
 * @return Calculated residual of the computation
 * 
 */
__global__ void poisson(int itermax, int imax, int jmax, double rdx2, double rdy2, int fluid_cells, double eps, double beta_2, double omega) {

    double p0 = 0.0;
    /* Calculate sum of squares */
    for (int i = 1; i < imax+1; i++) {
        for (int j = 1; j < jmax+1; j++) {
            if (cuda_flag[at(i, j)] & C_F) { p0 += cuda_p[at(i, j)] * cuda_p[at(i, j)]; }
        }
    }
   
    p0 = sqrt(p0 / fluid_cells); 
    if (p0 < 0.0001) { p0 = 1.0; }

    /* Red/Black SOR-iteration */
    int iter;
    double res = 0.0;
    for (iter = 0; iter < itermax; iter++) {


        for (int rb = 0; rb < 2; rb++) {

            for (int i = 1; i < imax+1; i++) {
                for (int j = 1; j < jmax+1; j++) {

                    if ((i + j) % 2 != rb) { continue; }

                    if (cuda_flag[at(i, j)] == (C_F | B_NSEW)) {
                        /* five point star for interior fluid cells */
                        cuda_p[at(i, j)] = (1.0 - omega) * cuda_p[at(i, j)] - 
                                beta_2 * ((cuda_p[at(i+1, j)] + cuda_p[at(i-1, j)] ) * rdx2
                                            + (cuda_p[at(i, j+1)] + cuda_p[at(i, j-1)]) * rdy2
                                            - cuda_rhs[at(i, j)]);

                    } else if (cuda_flag[at(i, j)] & C_F) { 
                        /* modified star near boundary */
                        double eps_E = ((cuda_flag[at(i+1, j)] & C_F) ? 1.0 : 0.0);
                        double eps_W = ((cuda_flag[at(i-1, j)] & C_F) ? 1.0 : 0.0);
                        double eps_N = ((cuda_flag[at(i, j+1)] & C_F) ? 1.0 : 0.0);
                        double eps_S = ((cuda_flag[at(i, j-1)] & C_F) ? 1.0 : 0.0);

                        double beta_mod = -omega / ((eps_E + eps_W) * rdx2 + (eps_N + eps_S) * rdy2);
                        cuda_p[at(i, j)] = (1.0 - omega) * cuda_p[at(i, j)] -
                            beta_mod * ((eps_E * cuda_p[at(i+1, j)] + eps_W * cuda_p[at(i-1, j)]) * rdx2
                                            + (eps_N * cuda_p[at(i, j+1)] + eps_S * cuda_p[at(i, j-1)]) * rdy2
                                            - cuda_rhs[at(i, j)]);
                    }
                }
            }
        }
        
        /* computation of residual */
        for (int i = 1; i < imax+1; i++) {
            for (int j = 1; j < jmax+1; j++) {
                if (cuda_flag[at(i, j)] & C_F) {
                    double eps_E = ((cuda_flag[at(i+1, j)] & C_F) ? 1.0 : 0.0);
                    double eps_W = ((cuda_flag[at(i-1, j)] & C_F) ? 1.0 : 0.0);
                    double eps_N = ((cuda_flag[at(i, j+1)] & C_F) ? 1.0 : 0.0);
                    double eps_S = ((cuda_flag[at(i, j-1)] & C_F) ? 1.0 : 0.0);

                    /* only fluid cells */
                    double add = (eps_E * (cuda_p[at(i+1, j)] - cuda_p[at(i, j)]) - 
                        eps_W * (cuda_p[at(i, j)] - cuda_p[at(i-1, j)])) * rdx2  +
                        (eps_N * (cuda_p[at(i, j+1)] - cuda_p[at(i, j)]) -
                         eps_S * (cuda_p[at(i, j)] - cuda_p[at(i, j-1)])) * rdy2  -  cuda_rhs[at(i, j)];
                    res += add * add;
                }
            }
        }
        res = sqrt(res / fluid_cells) / p0;
        
        /* convergence? */
        if (res < eps) break;
    }
}


/**
 * @brief Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
__global__ void update_velocity(int imax, int jmax, double delx, double dely, double del_t) {   
    for (int i = 1; i < imax-2; i++) {
        for (int j = 1; j < jmax-1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((cuda_flag[at(i, j)] & C_F) && (cuda_flag[at(i+1, j)] & C_F)) {
                cuda_u[at(i, j)] = cuda_f[at(i, j)] - (cuda_p[at(i+1, j)] - cuda_p[at(i, j)]) * del_t / delx;
            }
        }
    }
    
    for (int i = 1; i < imax-1; i++) {
        for (int j = 1; j < jmax-2; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((cuda_flag[at(i, j)] & C_F) && (cuda_flag[at(i, j+1)] & C_F)) {
                cuda_v[at(i, j)] = cuda_g[at(i, j)] - (cuda_p[at(i, j+1)] - cuda_p[at(i, j)]) * del_t / dely;
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
    double res, t, tv_time, rhs_time, p_time, v_time, boundary_time;
    return;

    /* Main loop */
    int iters = 0;
    for (t = 0.0; t < t_end; t += del_t, iters++) {
        if (!fixed_dt)
            set_timestep_interval();

        compute_tentative_velocity<<<1, 1>>>(imax, jmax, delx, dely, del_t, Re, y);
        cudaDeviceSynchronize();

        compute_rhs<<<1, 1>>>(imax, jmax, delx, dely, del_t);
        cudaDeviceSynchronize();

        poisson<<<1, 1>>>(itermax, imax, jmax, rdx2, rdy2, fluid_cells, eps, beta_2, omega);
        cudaDeviceSynchronize();

        update_velocity<<<1, 1>>>(imax, jmax, delx, dely, del_t);
        cudaDeviceSynchronize();

        apply_boundary_conditions<<<1, 1>>>(imax, jmax, ui, vi);
        cudaDeviceSynchronize();

        if ((iters % output_freq == 0)) {
            printf("Step %8d, Time: %14.8e (del_t: %14.8e), Residual: %14.8e\n", iters, t+del_t, del_t, res);
            print_timer("compute_tentative_velocity", tv_time);
            print_timer("compute_rhs", rhs_time);
            print_timer("poisson", p_time);
            print_timer("update_velocity", v_time);
            print_timer("apply_boundary_conditions", boundary_time);
            if(print_time)
                printf("\n");


            if ((!no_output) && (enable_checkpoints))
                write_checkpoint(iters, t+del_t);
        }
    } /* End of main loop */
    get_data();
    printf("Step %8d, Time: %14.8e, Residual: %14.8e\n", iters, t, res);
    printf("Simulation complete.\n");

    if (!no_output)
        write_result(iters, t);
}


/**
 * @brief The main routine that sets up the problem and executes the solving routines routines
 * 
 * @param argc The number of arguments passed to the program
 * @param argv An array of the arguments passed to the program
 * @return int The return value of the application
 */
int main(int argc, char *argv[]) {
    double setup_time, main_loop_time;
    int deviceId = 0;
    cudaDeviceProp prop;

    checkCuda( cudaSetDevice(deviceId) );
    checkCuda( cudaGetDeviceProperties(&prop, deviceId) );
    printf("Device: %s\n", prop.name);

    setup_time = get_time();
    set_defaults();
    parse_args(argc, argv);
    setup();

    if (verbose) print_opts();

    allocate_arrays();
    allocate_cuda_arrays();
    problem_set_up();
    test();
    setup_time = get_time() - setup_time;
    print_timer("Setup", setup_time);


    time(main_loop(), main_loop_time);
    print_timer("Main loop", main_loop_time);
    free_all();

    return 0;
}
