#ifndef BOUNDARY_H
#define BOUNDARY_H

__global__ void apply_boundary_conditions(double **u, double **v, char **flag, int imax, int jmax);

#endif