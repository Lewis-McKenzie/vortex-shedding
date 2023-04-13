#ifndef MPI_TOOLS_H
#define MPI_TOOLS_H

void sync_all();
void sync(void** target, MPI_Datatype datatype);
void broadcast(void** target, MPI_Datatype datatype);
void combine_2d_array(void** target, MPI_Datatype datatype);
void swap_edge_arrays(void** target, MPI_Datatype datatype);
void swap_right(void** target, MPI_Datatype datatype);
void swap_left(void** target, MPI_Datatype datatype);

#endif