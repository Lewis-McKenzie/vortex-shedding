#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#include "data.h"
#include "vtk.h"
#include "setup.h"
#include "boundary.h"
#include "args.h"
#include "mpi_tools.h"

struct timespec timer;
double get_time() {
	clock_gettime(CLOCK_MONOTONIC, &timer); 
	return (double) (timer.tv_sec + timer.tv_nsec / 1000000000.0);
}

#define time(func, timer) if(print_time){timer = get_time();func;timer = get_time() - timer;}else{func;}
#define print_timer(name, timer) if(print_time)printf("%s: %lf\n", name, timer);
#define init_outer_loop(index, limit, max) index = rank * (imax+2) / size; limit = (rank+1) * (imax+2) / size;if (rank == 0) {index = 1;}if (limit > max) {limit = max;}
/**
 * @brief Computation of tentative velocity field (f, g)
 * 
 */
void compute_tentative_velocity() {
    int i, i_limit;
    init_outer_loop(i, i_limit, imax);
    for (; i < i_limit; i++) {
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

    init_outer_loop(i, i_limit, imax+1);
    for (; i < i_limit; i++) {
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

    if (rank == 0) {
        /* f & g at external boundaries */
        for (int j = 1; j < jmax+1; j++) {
            f[0][j]    = u[0][j];
        }
    }
    if (rank == size - 1) {
        /* f & g at external boundaries */
        for (int j = 1; j < jmax+1; j++) {
            f[imax][j] = u[imax][j];
        }
    }

    init_outer_loop(i, i_limit, imax+1);
    for (; i < i_limit; i++) {
        g[i][0]    = v[i][0];
        g[i][jmax] = v[i][jmax];
    }
    //sync((void **)g, MPI_DOUBLE);
    //sync((void **)f, MPI_DOUBLE);
}


/**
 * @brief Calculate the right hand side of the pressure equation 
 * 
 */
void compute_rhs() {
    int i, i_limit;
    init_outer_loop(i, i_limit, imax+1);
    for (; i < i_limit; i++) {
        for (int j = 1;j < jmax+1; j++) {
            if (flag[i][j] & C_F) {
                /* only for fluid and non-surface cells */
                rhs[i][j] = ((f[i][j] - f[i-1][j]) / delx + 
                             (g[i][j] - g[i][j-1]) / dely)
                            / del_t;
            }
        }
    }
    //sync((void **)rhs, MPI_DOUBLE);
}


/**
 * @brief Red/Black SOR to solve the poisson equation.
 * 
 * @return Calculated residual of the computation
 * 
 */
double poisson() {
    double p0 = 0.0;
    /* Calculate sum of squares */
    int i, i_limit;
    init_outer_loop(i, i_limit, imax+1);
    for (; i < i_limit; i++) {
        for (int j = 1; j < jmax+1; j++) {
            if (flag[i][j] & C_F) { p0 += p[i][j] * p[i][j]; }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &p0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   
    p0 = sqrt(p0 / fluid_cells); 
    if (p0 < 0.0001) { p0 = 1.0; }

    /* Red/Black SOR-iteration */
    int iter;
    double res = 0.0;
    for (iter = 0; iter < itermax; iter++) {

        for (int rb = 0; rb < 2; rb++) {
            init_outer_loop(i, i_limit, imax+1);
            for (; i < i_limit; i++) {
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

        //sync((void **) p, MPI_DOUBLE);
        
        /* computation of residual */
        init_outer_loop(i, i_limit, imax+1);
        for (; i < i_limit; i++) {
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
        MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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
void update_velocity() {
    int i, i_limit;
    init_outer_loop(i, i_limit, imax-2);
    for (; i < i_limit; i++) {
        for (int j = 1; j < jmax-1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                u[i][j] = f[i][j] - (p[i+1][j] - p[i][j]) * del_t / delx;
            }
        }
    }

    init_outer_loop(i, i_limit, imax-1);
    for (; i < i_limit; i++) {
        for (int j = 1; j < jmax-2; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                v[i][j] = g[i][j] - (p[i][j+1] - p[i][j]) * del_t / dely;
            }
        }
    }
    //sync((void **) u, MPI_DOUBLE);
    //sync((void **) v, MPI_DOUBLE);
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
    double res, t, tv_time, rhs_time, p_time, v_time, boundary_time, sync_time;

    /* Main loop */
    int iters = 0;
    for (t = 0.0; t < t_end; t += del_t, iters++) {
        if (!fixed_dt)
            set_timestep_interval();

        time(compute_tentative_velocity(), tv_time);

        time(compute_rhs(), rhs_time);

        time(res = poisson(), p_time);

        time(update_velocity(), v_time);

        time(apply_boundary_conditions(), boundary_time);

        time(sync_all(), sync_time);

        if ((iters % output_freq == 0) && (rank == 0)) {
            printf("Step %8d, Time: %14.8e (del_t: %14.8e), Residual: %14.8e\n", iters, t+del_t, del_t, res);
            print_timer("compute_tentative_velocity", tv_time);
            print_timer("compute_rhs", rhs_time);
            print_timer("poisson", p_time);
            print_timer("update_velocity", v_time);
            print_timer("apply_boundary_conditions", boundary_time);
            print_timer("sync", sync_time);
            if(print_time)
                printf("\n");


            if ((!no_output) && (enable_checkpoints))
                write_checkpoint(iters, t+del_t);
        }
        // Average the time
        MPI_Allreduce(MPI_IN_PLACE, &t, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        t = t / size;

    } /* End of main loop */

    if (rank == 0) {
        printf("Step %8d, Time: %14.8e, Residual: %14.8e\n", iters, t, res);
        printf("Simulation complete.\n");

        if (!no_output)
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
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double setup_time, main_loop_time;

    setup_time = get_time();
    set_defaults();
    parse_args(argc, argv);
    setup();

    if (verbose) print_opts();

    allocate_arrays();
    problem_set_up();
    setup_time = get_time() - setup_time;
    if (rank == 0) {
        print_timer("Setup", setup_time);
    }

    time(main_loop(), main_loop_time);
    if (rank == 0) {
        print_timer("Main loop", main_loop_time);
    }

    free_arrays();

    MPI_Finalize();

    return 0;
}
