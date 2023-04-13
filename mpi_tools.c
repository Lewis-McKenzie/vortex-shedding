#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "mpi_tools.h"
#include "data.h"

void sync_all() {
    sync((void**) u, MPI_DOUBLE);
    sync((void**) v, MPI_DOUBLE);
    sync((void**) f, MPI_DOUBLE);
    sync((void**) g, MPI_DOUBLE);
    sync((void**) p, MPI_DOUBLE);
    sync((void**) rhs, MPI_DOUBLE);
}

void sync(void** target, MPI_Datatype datatype) {
    combine_2d_array(target, datatype);
    broadcast(target, datatype);
}


void alt(void** target, MPI_Datatype datatype) {
    int ptr = rank * imax / size;
    int count = imax / size;
    if (rank != 0) {
        ptr++;
    }
    if (rank == size - 1 || rank == 0) {
        count++;
    }
    
    MPI_Allgather(target[ptr], count, datatype, target[0], (imax+2) * (jmax+2), datatype, MPI_COMM_WORLD);
}

void broadcast(void** target, MPI_Datatype datatype) {
    check_mpi(MPI_Bcast(target[0], (imax+2) * (jmax+2), datatype, 0, MPI_COMM_WORLD));
}

void combine_2d_array(void** target, MPI_Datatype datatype) {
    if (rank == 0) {
        for (int r = 1; r < size; r++) {
            // index of the first row to recieve
            int ptr = r * imax / size + 1;
            // number of rows to get
            int count = imax / size;
            if (r == size - 1) {
                count++;
            }

            MPI_Status status;
            check_mpi(MPI_Recv(target[ptr], count * (jmax+2), datatype, r, MPI_ANY_TAG, MPI_COMM_WORLD, &status));

            //int c;
            //MPI_Get_count(&status, MPI_DOUBLE, &c); // how many doubles did we actually receive?
            //printf("I've received %d doubles from %d, with the tag %d\n", c, status.MPI_SOURCE, status.MPI_TAG);
        }
    } else {
        // index of the first row to send
        int ptr = rank * imax / size + 1;
        // number of rows to send
        int count = imax / size;        
        if (rank == size - 1) {
            count++;
        }

        check_mpi(MPI_Send(target[ptr], count * (jmax+2), datatype, 0, 0, MPI_COMM_WORLD));

    }
}

void swap_edge_arrays(void** target, MPI_Datatype datatype) {
    swap_right(target, datatype);
    swap_left(target, datatype);
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