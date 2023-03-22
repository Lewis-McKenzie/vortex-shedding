#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include "vtk.h"
#include "data.h"

#define DATA_CU

double xlength = 4.0;     /* Width of simulated domain */
double ylength = 1.0;     /* Height of simulated domain */
int imax = 512;           /* Number of cells horizontally */
int jmax = 128;           /* Number of cells vertically */

double t_end = 5.0;        /* Simulation runtime */
double del_t = 0.003;      /* Duration of each timestep */
double tau = 0.5;          /* Safety factor for timestep control */

int itermax = 100;         /* Maximum number of iterations in SOR */
double eps = 0.001;        /* Stopping error threshold for SOR */
double omega = 1.7;        /* Relaxation parameter for SOR */
double y = 0.9;            /* Gamma, Upwind differencing factor in PDE discretisation */

double Re = 500.0;         /* Reynolds number */
double ui = 1.0;           /* Initial X velocity */
double vi = 0.0;           /* Initial Y velocity */

double delx, dely;
double rdx2, rdy2;
double beta_2;

int fluid_cells = 0;

// Grids used for veclocities, pressure, rhs, flag and temporary f and g arrays
int u_size_x, u_size_y;
double ** u;
__device__ double ** cuda_u;
int v_size_x, v_size_y;
double ** v;
__device__ double ** cuda_v;
int p_size_x, p_size_y;
double ** p;
__device__ double ** cuda_p; 
int rhs_size_x, rhs_size_y;
double ** rhs;
__device__ double ** cuda_rhs;
int f_size_x, f_size_y;
double ** f;
int g_size_x, g_size_y;
double ** g;
int flag_size_x, flag_size_y;
char ** flag;
__device__ char ** cuda_flag;

/**
 * @brief Allocate a 2D array that is addressable using square brackets
 * 
 * @param m The first dimension of the array
 * @param n The second dimension of the array
 * @return double** A 2D array
 */
double **alloc_2d_array(int m, int n) {
  	double **x;
  	int i;

  	x = (double **)malloc(m*sizeof(double *));
  	x[0] = (double *)calloc(m*n,sizeof(double));
  	for ( i = 1; i < m; i++ )
    	x[i] = &x[0][i*n];
	return x;
}



/**
 * @brief Allocate a 2D char array that is addressable using square brackets
 * 
 * @param m The first dimension of the array
 * @param n The second dimension of the array
 * @return char** A 2D array
 */
char **alloc_2d_char_array(int m, int n) {
  	char **x;
  	int i;

  	x = (char **)malloc(m*sizeof(char *));
  	x[0] = (char *)calloc(m*n,sizeof(char));
  	for ( i = 1; i < m; i++ )
    	x[i] = &x[0][i*n];
	return x;
}

double **alloc_2d_cuda_array(int m, int n) {
  	double **x;
  	int i;

  	cudaMalloc((void**) x, m*sizeof(double *));
  	cudaMalloc((void**) x[0], m*n*sizeof(double));
  	for ( i = 1; i < m; i++ )
    	x[i] = &x[0][i*n];
	return x;
}

char **alloc_2d_char_cuda_array(int m, int n) {
  	char **x;
  	int i;

  	cudaMalloc((void**) x, m*sizeof(char *));
  	cudaMalloc((void**) x[0], m*n*sizeof(char));
  	for ( i = 1; i < m; i++ )
    	x[i] = &x[0][i*n];
	return x;
}

void to_gpu_2d(void** array, void** cuda_array, int m, int size) {
	for (int i = 0; i < m; i++) {
		cudaMemcpy(cuda_array[i], array[i], size, cudaMemcpyHostToDevice);
	}
}

void from_gpu_2d(void** array, void** cuda_array, int m, int size) {
	for (int i = 0; i < m; i++) {
		cudaMemcpy(array[i], cuda_array[i], size, cudaMemcpyDeviceToHost);
	}
}

/**
 * @brief Free a 2D array
 * 
 * @param array The 2D array to free
 */
void free_2d_array(void ** array) {
	free(array[0]);
	free(array);
}

void free_2d_cuda_array(void ** array) {
	cudaFree(array[0]);
	cudaFree(array);
}
