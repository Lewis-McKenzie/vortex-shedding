#ifndef BOUNDARY_H
#define BOUNDARY_H

__device__ void apply_boundary_conditions(double **u, double **v, char **flag, int imax, int jmax, double ui, double vi);

#endif