#ifndef DATA_H
#define DATA_H

#define C_B      0x0000   /* This cell is an obstacle/boundary cell */
#define B_N      0x0001   /* This obstacle cell has a fluid cell to the north */
#define B_S      0x0002   /* This obstacle cell has a fluid cell to the south */
#define B_W      0x0004   /* This obstacle cell has a fluid cell to the west */
#define B_E      0x0008   /* This obstacle cell has a fluid cell to the east */
#define B_NW     (B_N | B_W)
#define B_SW     (B_S | B_W)
#define B_NE     (B_N | B_E)
#define B_SE     (B_S | B_E)
#define B_NSEW   (B_N | B_S | B_E | B_W)

#define C_F      0x0010    /* This cell is a fluid cell */

extern __device__ double xlength;     /* Width of simulated domain */
extern __device__ double ylength;     /* Height of simulated domain */
extern int imax;           /* Number of cells horizontally */
extern int jmax;           /* Number of cells vertically */

extern double t_end;       /* Simulation runtime */
extern double del_t;       /* Duration of each timestep */
extern __device__ double tau;         /* Safety factor for timestep control */

extern __device__ int itermax;        /* Maximum number of iterations in SOR */
extern __device__ double eps;         /* Stopping error threshold for SOR */
extern __device__ double omega;       /* Relaxation parameter for SOR */
extern __device__ double y;           /* Gamma, Upwind differencing factor in PDE */

extern __device__ double Re;          /* Reynolds number */
extern __device__ double ui;          /* Initial X velocity */
extern __device__ double vi;          /* Initial Y velocity */

extern __device__ int fluid_cells;

extern __device__ double delx, dely;
extern __device__ double rdx2, rdy2;
extern __device__ double beta_2;

extern int block_dim;
extern int grid_dim;

extern double *reduction_buffer;

// Grids used for veclocities, pressure, rhs, flag and temporary f and g arrays
extern int u_size_x, u_size_y;
extern double ** u;
extern int v_size_x, v_size_y;
extern double ** v;
extern int p_size_x, p_size_y;
extern double ** p; 
extern int rhs_size_x, rhs_size_y;
extern double ** rhs; 
extern int f_size_x, f_size_y;
extern double ** f; 
extern int g_size_x, g_size_y;
extern double ** g;
extern int flag_size_x, flag_size_y;
extern char ** flag;

__global__ void test();

cudaError_t checkCuda(cudaError_t result);
double **alloc_2d_array(int m, int n);
char **alloc_2d_char_array(int m, int n);
void free_2d_array(void ** array);

#endif