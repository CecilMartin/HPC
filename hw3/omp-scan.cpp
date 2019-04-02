#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  //Here I use inclusive scan to be consistent with probelm discreption.
  if (n == 0) return;
  prefix_sum[0] = A[0];
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan\int nthreads = 8;
  int nthreads = 4;
  //Assume that n//nthreads
  int N_thread = n / nthreads;
  printf("Max threads in this machine is %d\n",omp_get_max_threads());
  long* batch = (long*) malloc(nthreads * sizeof(long));
  long* prefix_sum_batch = (long*) malloc(nthreads * sizeof(long));
  omp_set_num_threads(nthreads);
  #pragma omp parallel for
  for (int j = 0; j < nthreads; j++){
    scan_seq(&prefix_sum[j*N_thread], &A[j*N_thread], N_thread);
    batch[j] = prefix_sum[(j+1)*N_thread - 1];
  }
  #pragma omp barrier
  scan_seq(prefix_sum_batch, batch, nthreads);
  #pragma omp parallel for
  for (int j = 1; j < nthreads; j++){
    for (int k = 0; k < N_thread; k++){
      prefix_sum[j*N_thread + k] += prefix_sum_batch[j-1];
    }
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
