#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

double time_ring_comm(long Nrepeat, long Nsize, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  if (size!=2) printf("Error! Use 2 procs\n");
  int* msg = (int*) malloc(Nsize*sizeof(int));
  for (long i = 0; i < Nsize; i++) msg[i] = 0;
  MPI_Barrier(comm);
  double tt = MPI_Wtime();
  int proc0 = 0, proc1 = 1;
  printf("I'm proc %i\n",rank);
  for (long repeat  = 0; repeat < Nrepeat; repeat++) {
    //if (!rank) printf("repeat=%i\n",repeat);
    MPI_Status status;
    if (repeat % 2 == 0) { // even iterations

      if (rank == proc0){
        msg[0] += rank;
        MPI_Send(msg, Nsize, MPI_INT, proc1, repeat, comm);
      }
      else if (rank == proc1){
        MPI_Recv(msg, Nsize, MPI_INT, proc0, repeat, comm, &status);
        //printf("Message %i received by proc %i in repeat %i\n",msg[0],rank,repeat );
      }
    }
    else { // odd iterations

      if (rank == proc0){
        MPI_Recv(msg, Nsize, MPI_INT, proc1, repeat, comm, &status);
        //printf("Message %i received by proc %i in repeat %i\n",msg[0],rank,repeat );
      }
      else if (rank == proc1){
        msg[0] += rank;
        MPI_Send(msg, Nsize, MPI_INT, proc0, repeat, comm);
      }
    }
  }
  tt = MPI_Wtime() - tt;
  if (Nrepeat % 2 == 0){
    if (rank == proc0) {
      printf("Result of msg[0] is %d while it should be %ld\n",msg[0],(Nrepeat/2) );
    }
  else{
    if (rank == proc1) {
      printf("Result of msg[0] is %d while it should be %ld\n",msg[0],(Nrepeat/2) );
    }
  }
  }


  free(msg);
  return tt;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  if (argc < 2) {
    printf("Usage: mpirun -np -2 ./int_ring <Nrepeat>\n");
    abort();
  }
  long Nrepeat = atol(argv[1]);

  int rank;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);

  double tt = time_ring_comm(Nrepeat, 1, comm);
  if (!rank) printf("pingpong latency: %e ms\n", tt/Nrepeat * 1000);
  MPI_Barrier(comm);
  if (!rank) printf("------------------------------------------------------------------------------\n");
  //ENrepeat = 1000;
  long Nsize = 2e6 / sizeof(int) ;
  tt = time_ring_comm(Nrepeat, Nsize, comm);
  MPI_Barrier(comm);
  if (!rank) printf("------------------------------------------------------------------------------\n");
  if (!rank) printf("pingpong bandwidth: %e GB/s\n", (Nsize*sizeof(int)*Nrepeat)/tt/1e9);

  MPI_Finalize();
}
