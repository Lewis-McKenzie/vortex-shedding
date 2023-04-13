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

void broadcast(void** target, MPI_Datatype datatype) {
    MPI_Bcast(target[0], (imax+2) * (jmax+2), datatype, 0, MPI_COMM_WORLD);
}

void combine_2d_array(void** target, MPI_Datatype datatype) {
    if (rank == 0) {
        for (int r = 1; r < size; r++) {
            // index of the first row to recieve
            int ptr = r * imax / size + 1;
            // number of rows to get
            int count = imax / size;
            if (r == size - 1) {
                count = (imax+1) - r * count;
            }

            MPI_Status status;
            MPI_Recv(target[ptr], count * (jmax+2), datatype, r, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
    } else {
        // index of the first row to send
        int ptr = rank * imax / size + 1;
        // number of rows to send
        int count = imax / size;        
        if (rank == size - 1) {
            count = (imax+1) - rank * count;
        }

        MPI_Send(target[ptr], count * (jmax+2), datatype, 0, 0, MPI_COMM_WORLD);

    }
}