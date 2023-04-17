#ifndef MPI_TOOLS_H
#define MPI_TOOLS_H

extern double io_bound_time;

void sync_all();
void sync(void** target, MPI_Datatype datatype);
void swap_edge_arrays(void** target, MPI_Datatype datatype);
void swap_right(void** target, MPI_Datatype datatype);
void swap_left(void** target, MPI_Datatype datatype);

#endif