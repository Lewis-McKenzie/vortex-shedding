#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>

#include "vtk.cuh"

__device__ double xlength = 4.0;     			  /* Width of simulated domain */
__device__ double ylength = 1.0;     			  /* Height of simulated domain */
int imax = 512;           			  /* Number of cells horizontally */
int jmax = 128;           			  /* Number of cells vertically */

double t_end = 5.0;        			  /* Simulation runtime */
double del_t = 0.003;      			  /* Duration of each timestep */
__device__ double tau = 0.5;          			  /* Safety factor for timestep control */

__device__ int itermax = 100;         /* Maximum number of iterations in SOR */
__device__ double eps = 0.001;        /* Stopping error threshold for SOR */
__device__ double omega = 1.7;        			  /* Relaxation parameter for SOR */
__device__ double y = 0.9;            			  /* Gamma, Upwind differencing factor in PDE discretisation */

__device__ double Re = 500.0;         /* Reynolds number */
__device__ double ui = 1.0;           			  /* Initial X velocity */
__device__ double vi = 0.0;           			  /* Initial Y velocity */

__device__ double delx, dely;
__device__ double rdx2, rdy2;
__device__ double beta_2;

__device__ int fluid_cells = 0;

int block_dim = 512;
int grid_dim = 1;

double *reduction_buffer;

// Grids used for veclocities, pressure, rhs, flag and temporary f and g arrays
int u_size_x, u_size_y;
double ** u;
int v_size_x, v_size_y;
double ** v;
int p_size_x, p_size_y;
double ** p; 
int rhs_size_x, rhs_size_y;
double ** rhs; 
int f_size_x, f_size_y;
double ** f; 
int g_size_x, g_size_y;
double ** g;
int flag_size_x, flag_size_y;
char ** flag;

__global__ void test() {
	printf("Here\n");
}

cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

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

	checkCuda(cudaMallocManaged(&x, m * sizeof(double*)));
	checkCuda(cudaMallocManaged(&x[0], m * n * sizeof(double)));

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

	checkCuda(cudaMallocManaged(&x, m * sizeof(char*)));
	checkCuda(cudaMallocManaged(&x[0], m * n * sizeof(char)));
  	for ( i = 1; i < m; i++ )
    	x[i] = &x[0][i*n];
	return x;
}

/**
 * @brief Free a 2D array
 * 
 * @param array The 2D array to free
 */
void free_2d_array(void ** array) {
	cudaFree(array[0]);
	cudaFree(array);
}
