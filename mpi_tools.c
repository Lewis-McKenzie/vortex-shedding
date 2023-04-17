#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

#include "mpi_tools.h"
#include "data.h"
#include "args.h"

#define time(func, timer) if(print_time){double tim = get_time();func;tim = get_time() - tim; timer += tim;}else{func;}

double io_bound_time;

void sync_all() {
    sync((void**) u, MPI_DOUBLE);
    sync((void**) v, MPI_DOUBLE);
    sync((void**) f, MPI_DOUBLE);
    sync((void**) g, MPI_DOUBLE);
    sync((void**) p, MPI_DOUBLE);
    sync((void**) rhs, MPI_DOUBLE);
}

void sync(void** target, MPI_Datatype datatype) {
    int ptr = rank * imax / size;
    int count = imax / size;
    if (rank != 0) {
        ptr++;
    }
    if (rank == size - 1 || rank == 0) {
        count++;
    }


    int *revscount = malloc(sizeof(int)*size);
    int *displ = malloc(sizeof(int)*size);
    for (int i = 0; i < size; i++) {
        revscount[i] = (imax/size)*(jmax+2);
        displ[i] = (i * imax / size + 1) * (jmax+2);
    }
    revscount[0] += jmax+2;
    revscount[size-1] += jmax+2;
    displ[0] = 0;
    
    MPI_Allgatherv(MPI_IN_PLACE, count*(jmax+2), datatype, target[0], revscount, displ, datatype, MPI_COMM_WORLD);

}

void swap_edge_arrays(void** target, MPI_Datatype datatype) {
    time((swap_right(target, datatype)), io_bound_time);
    time((swap_left(target, datatype)), io_bound_time);
}

void swap_right(void** target, MPI_Datatype datatype) {
    int ptr = rank * imax / size + 1;
    int count = imax / size;
    // send rightmost column
    if (rank != size - 1) {
        MPI_Send(target[ptr+count-1], jmax+2, datatype, rank+1, 0, MPI_COMM_WORLD);
    }
    //recieve leftmost column
    if (rank != 0) {
        MPI_Status status;
        MPI_Recv(target[ptr-1], jmax+2, datatype, rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
}

void swap_left(void** target, MPI_Datatype datatype) {
    int ptr = rank * imax / size + 1;
    int count = imax / size;
    // send leftmost column
    if (rank != 0) {
        MPI_Send(target[ptr], jmax+2, datatype, rank-1, 0, MPI_COMM_WORLD);
    }
    //recieve rightmost column
    if (rank != size-1) {
        MPI_Status status;
        MPI_Recv(target[ptr+count], jmax+2, datatype, rank+1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
}
