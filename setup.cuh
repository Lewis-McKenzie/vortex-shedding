#ifndef SETUP_H
#define SETUP_H

void set_defaults();
void cuda_setup(int imax, int jmax, double delx, double dely);
void setup();
void allocate_arrays();
void free_arrays();
__global__ void cuda_arr_setup(double **u, double **v, double **p, int imax, int jmax);
void problem_set_up();

#endif