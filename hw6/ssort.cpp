// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  
  //Debug mode!!!!!!!!!!!!!!!!!!!!!
  // if (rank==0){
  //   int i = 0;
  //   char hostname[256];
  //   gethostname(hostname, sizeof(hostname));
  //   printf("PID %d on %s ready for attach\n", getpid(), hostname);
  //   fflush(stdout);
  //   while (0 == i)
  //       sleep(5);
  // }
  // printf("Hello rank %d\n",rank );

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N=1024;
  sscanf(argv[1], "%d", &N);


  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core

  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
      vec[i] = rand();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  printf("rank: %d, first entry: %d\n", rank, vec[0]);

  double tt = MPI_Wtime();

  // sort locally
  std::sort(vec, vec+N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int* local_sp = (int*)malloc((p-1)*sizeof(int));
  for (int i = 0; i < p-1; i++) {
    local_sp[i] = vec[(i+1)*(N/p)-1];
  }

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  int* root_sp = NULL;
  if (rank == 0){
    root_sp = (int*)malloc(sizeof(int)*(p-1)*p);
  }
  MPI_Gather(local_sp,p-1,MPI_INT,root_sp,p-1,MPI_INT,0,MPI_COMM_WORLD);

  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  int* global_sp = (int*)malloc((p-1)*sizeof(int));
  if (rank == 0){
    std::sort(root_sp,root_sp+(p-1)*p);
    for (int i = 0; i < p-1; i++){
      global_sp[i]=root_sp[i*p+(p-1)/2];
    }
  }

  // root process broadcasts splitters to all other processes
  MPI_Bcast(global_sp,p-1,MPI_INT,0,MPI_COMM_WORLD);

  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;
  int* sdispls = (int*)malloc((p)*sizeof(int));
  int* sendcnts = (int*)malloc((p)*sizeof(int));
  sdispls[0]=0;
  for (int i=0; i < p-1; i++){
  	sdispls[i+1]=std::lower_bound(vec, vec+N, global_sp[i])-vec;
  	sendcnts[i] = sdispls[i+1]-sdispls[i];
  }
  sendcnts[p-1]=N-sdispls[p-1];


  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data
  int * recvcnts = (int*)malloc((p)*sizeof(int));
  MPI_Alltoall(sendcnts, 1, MPI_INT,recvcnts, 1, MPI_INT,MPI_COMM_WORLD);//max recv number
  int * rdispls = (int*) malloc(p*sizeof(int));
  rdispls[0]=0;
  for (int i=1; i < p; i++){
  	rdispls[i]=rdispls[i-1]+recvcnts[i-1];
  }
  int local_size = rdispls[p-1]+recvcnts[p-1];//size of this recvbuf
  int* recvbuf = (int*)malloc(local_size*sizeof(int));
  MPI_Alltoallv(vec, sendcnts, sdispls, MPI_INT, recvbuf, recvcnts, rdispls, MPI_INT, MPI_COMM_WORLD);

  // do a local sort of the received data
  std::sort(recvbuf,recvbuf+local_size);

  // every process writes its result to a file
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0){
    printf("Total time is %f (s)\n",MPI_Wtime()-tt);
  }
  { // Write output to a file
    FILE* fd = NULL;
    char filename[256];
    snprintf(filename, 256, "output%02d.txt", rank);
    fd = fopen(filename,"w+");

    if(NULL == fd) {
      printf("Error opening file \n");
      return 1;
    }

    fprintf(fd, "//rank %d\n", rank);
    for(int i = 0; i < local_size; i++)
      fprintf(fd, "%d\n", recvbuf[i]);

    fclose(fd);
  }

  //free space
  free(root_sp);
  free(vec);
  free(global_sp);
  free(sdispls);
  free(sendcnts);
  free(recvcnts);
  free(rdispls);
  free(recvbuf);
  free(local_sp);
  MPI_Finalize();
  return 0;
}
