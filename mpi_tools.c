#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "mpi_tools.h"
#include "data.h"

void broadcast() {

}

void combine_2d_array(void** target, int type_size, MPI_Datatype datatype) {
    if (rank == 0) {
        for (int r = 1; r < size; r++) {
            // index of the first row to recieve
            int ptr = r * imax / size;
            // number of rows to get
            int count = imax / size;
            if (r == size - 1) {
                count = imax - r * count;
            }

            void* buffer = malloc(type_size * count * jmax);
            MPI_Status status;
            MPI_Recv(buffer, count * jmax, datatype, r, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            memcpy(target[ptr], buffer, type_size * count * jmax);
        }
    } else {
        // index of the first row to send
        int ptr = rank * imax / size;
        // number of rows to send
        int count = imax / size;        
        if (rank == size - 1) {
            count = imax - rank * count;
        }

        void* buffer = malloc(type_size * count * jmax);
        memcpy(buffer, target[ptr], type_size * count * jmax);
        MPI_Send(buffer, count * jmax, datatype, 0, 0, MPI_COMM_WORLD);

    }
}