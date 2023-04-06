#ifndef SETUP_H
#define SETUP_H

void set_defaults();
__global__ void setup(int imax, int jmax);
void allocate_arrays();
void free_arrays();
__global__ void problem_set_up(double **u, double **v, double **p, char ** flag, int imax, int jmax) ;

#endif