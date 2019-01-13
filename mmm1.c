#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <limits.h>
#include <assert.h>


// Data needs to be sent to each process in a block cyclic manner
// Do the data reordering for that
void reorderDataToBeSent(
        double         **matA,
        int            A_row_size,
        double         **matB,
        int            B_col_size,
        double         **bufA,
        double         **bufB,
        int            world_size,
        int            world_rank,
        int            row_size,
        int            col_size)
{
    if (world_rank != 0)
        return;
    (*bufA) = (double*) memalign(64, sizeof(double)*A_row_size);
    (*bufB) = (double*) memalign(64, sizeof(double)*B_col_size);

    // It is fairly simple to order A
    // every processor gets M / world_size elements in a cyclic manner
    int count = 0;
    for (int i = 0; i < world_size; ++i) {
        for (int j = i; j < A_row_size; j += world_size) {
            (*bufA)[count++] = (*matA)[j];
        }
    }

    count = 0;
    int max_size = B_col_size;
    for (int i = 0; i < row_size; ++i) {
        for (int j = 0; j < col_size; ++j) {
            int start = i + j * row_size;
            while (start < max_size) {
                (*bufB)[count++] = (*matB)[start];
                start += world_size;
            }
        }
    }
}


// Initialize matrices // just vectors for part A
void initMatrices(
        double        **matA,
        double        **matB,
        int           A_row_size,
        int           B_col_size,
        int           world_rank)
{
    if (world_rank != 0)
        return;
    (*matA) = (double*) memalign(64, sizeof(double)*A_row_size);
    (*matB) = (double*) memalign(64, sizeof(double)*B_col_size);
    printf("Vector A\n");
    for (int i = 0; i < A_row_size; ++i)  {
        (*matA)[i] = i + 1;
        printf("%lf ", (*matA)[i]);
    }
    printf("\n\n");
    printf("Vector B\n");
    for (int i = 0; i < B_col_size; ++i)  {
        (*matB)[i] = i + 1;
        printf("%lf ", (*matB)[i]);
    }
    printf("\n\n");
}

// After scattering the elements
//         _______________
//        |   A0  |   A1  |
//        |B0     |B2     |
//        |   P1  |   P2  |
//        |_______|_______|
//        |   A2  |   A3  |
//        |B1     |B3     |
//        |   P3  |   P4  |
//        |_______|_______|

// After all gather in rows and columns
//
//         _______________
//        | A0 A1 | A0 A1 |
//        |B0     |B2     |
//        |B1     |B3     |
//        |_______|_______|
//        | A2 A3 | A2 A3 |
//        |B0     |B2     |
//        |B1     |B3     |
//        |_______|_______|
//
//
// After outer product
//
//         _______________
//        | A0B0  |  A0B2 |
//        | A0B1  |  A0B3 |
//        | A1B0  |  A1B2 |
//        |_A1B1__|__A1B3_|
//        | A2B0  |  A2B2 |
//        | A2B1  |  A2B3 |
//        | A3B0  |  A3B2 |
//        |_A3B1__|__A3B3_|
//
// Simple computation of the outer product
void initializeAndComputeProd(
        int            A_row_size,
        int            B_col_size,
        MPI_Comm       row_comm,
        MPI_Comm       col_comm,
        int            world_rank,
        int            world_size,
        int            row_size,
        int            col_size,
        int            row_rank,
        int            col_rank)
{
    // Allowing A and B vectors as multiples of number of processors
    assert (A_row_size % world_size == 0);
    assert (B_col_size % world_size == 0);

    double *matA = NULL, *matB = NULL;
    double *bufA = NULL, *bufB = NULL;

    initMatrices(&matA, &matB, A_row_size, B_col_size, world_rank);
    reorderDataToBeSent(&matA, A_row_size, &matB, B_col_size, &bufA, &bufB,
            world_size, world_rank, row_size, col_size);

    int r_size = A_row_size / world_size * row_size;
    int c_size = B_col_size / world_size * col_size;

    double *myAVec = (double*) memalign(64, sizeof(double)*A_row_size / world_size);
    double *myBVec = (double*) memalign(64, sizeof(double)*B_col_size / world_size);
    double *row_gather = (double*) memalign(64, sizeof(double)*r_size);
    double *col_gather = (double*) memalign(64, sizeof(double)*c_size);
    double *C = (double*) memalign(64, sizeof(double)*r_size*c_size);

    // Scatter the data to all processors
    MPI_Scatter(bufA, A_row_size / world_size, MPI_DOUBLE,
            myAVec, A_row_size / world_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(bufB, B_col_size  / world_size, MPI_DOUBLE,
            myBVec, B_col_size / world_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Do an all gather in the rows
    MPI_Allgather(myAVec, A_row_size / world_size, MPI_DOUBLE, row_gather,
            A_row_size / world_size, MPI_DOUBLE, row_comm);

    // Do an all gather in the columns
    MPI_Allgather(myBVec, B_col_size / world_size, MPI_DOUBLE, col_gather,
            B_col_size / world_size, MPI_DOUBLE, col_comm);

    // Compute outer product
    for (int i = 0; i < r_size; ++i) {
        for (int j = 0; j < c_size; ++j) {
            C[j + i * c_size] = row_gather[i] * col_gather[j];
        }
    }

    if (world_rank == 0) {
        printf("Local C for processor 0\n");
        for (int i = 0; i < r_size; ++i) {
            for (int j = 0; j < c_size; ++j) {
                printf("%lf ", C[j + i * c_size]);
            }
            printf("\n");
        }
    }

    free(matA); free(matB); free(bufA); free(bufB);
    free(myAVec); free(myBVec); free(row_gather);
    free(col_gather); free(C);
}


// the first argument to the main function specifies how many processes are in the row

int main(int argc, char** argv) {
    // initialize MPI
    MPI_Init(&argc, &argv);

    int row_elements = atoi(argv[1]);
    // Store the transpose of A and B
    int A_row_size = atoi(argv[2]);
    int B_col_size = atoi(argv[3]);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int row_color = world_rank / row_elements;
    int col_color = world_rank % row_elements;

    //create row and column communicator based on colors
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, world_rank, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col_color, world_rank, &col_comm);

    //find out row and column rank and sizes;
    int row_rank, row_size, col_rank, col_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    printf("World Rank/Size: %d/%d\t Row Rank/Size: %d/%d\t Col Rank/Size: %d/%d\n", world_rank, world_size, row_rank, row_size, col_rank, col_size);

    initializeAndComputeProd(A_row_size, B_col_size, row_comm, col_comm,
            world_rank, world_size, row_size, col_size, row_rank, col_rank);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    //clean up code for MPI
    MPI_Finalize();
    return (0);
}
