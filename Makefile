compile:
	make clean
	mpicc reduce_short.c -o reduce_short.x -std=c99
	mpicc allgather_long.c -o allgather_long.x -std=c99
	mpicc reduce_long.c -o reduce_long.x -std=c99
	mpicc allgather_short.c -o allgather_short.x -std=c99
	mpicc mmm1.c -o mmm1.x -std=c99
	mpicc mmm2.c -o mmm2.x -std=c99
	mpicc mmm3.c -o mmm3.x -std=c99

coll_comm:
	mpiexec -n 5 ./reduce_short.x 15 2
	mpiexec -n 5 ./allgather_long.x 20
	mpiexec -n 5 ./reduce_long.x 15 4 
	mpiexec -n 4 ./allgather_short.x 16

matrix_mul:
	mpiexec -n 4 ./mmm1.x 1 4 8
	mpiexec -n 5 ./mmm2.x 5 5 5 10
	mpiexec -n 5 ./mmm3.x 5 5 5 10

clean:
	rm -rf *.x

test_split:
	mpicc mpi_row_col.c -o mpi_row_col.x
	mpiexec -n 16 ./mpi_row_col.x 8
