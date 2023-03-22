#include <stdio.h>
#include <stdlib.h>

#include "data.h"
#include "vtk.h"
#include "boundary.h"

/**
 * @brief Set up some default values before arguments are parsed.
 * 
 */
void set_defaults() {
	set_default_base();
}

/**
 * @brief Set up some values after arguments have been parsed.
 * 
 */
void setup() {
	delx = xlength/imax;
    dely = ylength/jmax;
    rdx2 = 1.0 / (delx * delx);
    rdy2 = 1.0 / (dely * dely);
    beta_2 = -omega / (2.0 * (rdx2 + rdy2));
}
 
/**
 * @brief Allocate all of the arrays used by the computation.
 * 
 */
void allocate_arrays() {
	    /* Allocate arrays */
    u_size_x = imax+2; u_size_y = jmax+2;
    u = alloc_2d_array(u_size_x, u_size_y);
    v_size_x = imax+2; v_size_y = jmax+2;
    v = alloc_2d_array(v_size_x, v_size_y);
    f_size_x = imax+2; f_size_y = jmax+2;
    f = alloc_2d_array(f_size_x, f_size_y);
    g_size_x = imax+2; g_size_y = jmax+2;
    g = alloc_2d_array(g_size_x, g_size_y);
    p_size_x = imax+2; p_size_y = jmax+2;
    p = alloc_2d_array(p_size_x, p_size_y);
    rhs_size_x = imax+2; rhs_size_y = jmax+2;
    rhs = alloc_2d_array(rhs_size_x, rhs_size_y);
    flag_size_x = imax+2; flag_size_y = jmax+2;
    flag = alloc_2d_char_array(flag_size_x, flag_size_y);

    if (!u || !v || !f || !g || !p || !rhs || !flag) {
        fprintf(stderr, "Couldn't allocate memory for matrices.\n");
		exit(1);
    }
}

void allocate_cuda_arrays() {
    cuda_u = alloc_2d_cuda_array(u_size_x, u_size_y);
    cuda_v = alloc_2d_cuda_array(v_size_x, v_size_y);
    cuda_p = alloc_2d_cuda_array(p_size_x, p_size_y);
    cuda_flag = alloc_2d_char_cuda_array(flag_size_x, flag_size_y);
    cuda_rhs = alloc_2d_cuda_array(rhs_size_x, rhs_size_y);
}

/**
 * @brief Free all of the arrays used for the computation.
 * 
 */
void free_arrays() {
	free_2d_array((void**) u);
    free_2d_array((void**) v);
    free_2d_array((void**) f);
    free_2d_array((void**) g);
    free_2d_array((void**) p);
    free_2d_array((void**) rhs);
    free_2d_array((void**) flag);
}

void free_cuda_arrays() {
    free_2d_cuda_array((void**) cuda_u);
    free_2d_cuda_array((void**) cuda_v);
    free_2d_cuda_array((void**) cuda_p);
    free_2d_cuda_array((void**) cuda_flag);
    free_2d_cuda_array((void**) cuda_rhs);
}

void free_all() {
    free_arrays();
    free_cuda_arrays();
}


/**
 * @brief Initialise the velocity arrays and then initialize the flag array, 
 * marking any obstacle cells and the edge cells as boundaries. The cells 
 * adjacent to boundary cells have their relevant flags set too.
 */
void problem_set_up() {
    for (int i = 0; i < imax+2; i++) {
        for (int j = 0; j < jmax+2; j++) {
            u[i][j] = ui;
            v[i][j] = vi;
            p[i][j] = 0.0;
        }
    }

    /* Mark a circular obstacle as boundary cells, the rest as fluid */
    double mx = 20.0 / 41.0 * jmax * dely;
    double my = mx;
    double rad1 = 5.0 / 41.0 * jmax * dely;
    for (int i = 1; i <= imax; i++) {
        for (int j = 1; j <= jmax; j++) {
            double x = (i - 0.5) * delx - mx;
            double y = (j - 0.5) * dely - my;
            flag[i][j] = (x*x + y*y <= rad1*rad1) ? C_B : C_F;
        }
    }
    
    /* Mark the north & south boundary cells */
    for (int i = 0; i <= imax + 1; i++) {
        flag[i][0]      = C_B;
        flag[i][jmax+1] = C_B;
    }
    /* Mark the east and west boundary cells */
    for (int j = 1; j <= jmax; j++) {
        flag[0][j]      = C_B;
        flag[imax+1][j] = C_B;
    }

    fluid_cells = imax * jmax;

    /* flags for boundary cells */
    for (int i = 1; i <= imax; i++) {
        for (int j = 1; j <= jmax; j++) {
            if (!(flag[i][j] & C_F)) {
                fluid_cells--;
                if (flag[i-1][j] & C_F) flag[i][j] |= B_W;
                if (flag[i+1][j] & C_F) flag[i][j] |= B_E;
                if (flag[i][j-1] & C_F) flag[i][j] |= B_S;
                if (flag[i][j+1] & C_F) flag[i][j] |= B_N;
            }
        }
    }

	apply_boundary_conditions();
    to_gpu_2d((void**) u, (void**) cuda_u, u_size_x, sizeof(double) * u_size_y);
    to_gpu_2d((void**) v, (void**) cuda_v, v_size_x, sizeof(double) * v_size_y);
    to_gpu_2d((void**) p, (void**) cuda_p, p_size_x, sizeof(double) * p_size_y);
    to_gpu_2d((void**) flag, (void**) cuda_flag, flag_size_x, sizeof(char) * flag_size_y);
    to_gpu_2d((void**) rhs, (void**) cuda_rhs, rhs_size_x, sizeof(double) * rhs_size_y);
}