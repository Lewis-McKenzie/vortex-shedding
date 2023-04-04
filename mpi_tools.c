#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "mpi_tools.h"
#include "data.h"

void broadcast() {

}

void combine_array(void* target, int type_size, int max, MPI_Datatype datatype) {
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            int ptr = rank * imax / size;
            int count = max / size;
            if (rank == size - 1) {
                count = max - rank * count;
            }

            void* buffer = malloc(type_size * count);
            MPI_Status status;
            MPI_Recv(buffer, count, datatype, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            memcpy(target + ptr, buffer, type_size * count);
        }
    } else {
        int count = max / size;
        int ptr = rank * imax / size;
        if (rank == size - 1) {
            count = max - rank * count;
        }


        void* buffer = malloc(type_size * count);
        memcpy(buffer, target + ptr, type_size * count);
        MPI_Send(buffer, count, datatype, 0, MPI_ANY_TAG, MPI_COMM_WORLD);

    }
}