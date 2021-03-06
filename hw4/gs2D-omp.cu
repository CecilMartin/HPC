//Zhe Chen
#include <iostream>
#include <cmath>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "utils.h"

using namespace std;

#define BLOCK_DIM 32
#define BLOCK_DIM_IN 30

double Residual(int N, double *U, double *F){
  double h=1.0/(N+1.0);
  double res=0.0, res_1=0.0;
  #pragma omp parallel for shared(U,F) private(res_1)\
  reduction(+:res)
  for (int j=1;j<=N;j++){
    for (int i=1;i<=N;i++){
      res_1=(-U[(N+2)*j+i-1]-U[(N+2)*(j-1)+i]-U[(N+2)*j+i+1]-U[(N+2)*(j+1)+i]+4.0*U[(N+2)*j+i])/h/h-F[(N+2)*j+i];
      res+=res_1*res_1;
    }
  }
  res=sqrt(res);
  return res;
}

void gs2D_cpu(int N, double *U, double *F, int maxit, int num_threads){
	//black red points version of Gaiss-Seidel algorithm.
	#if defined(_OPENMP)
	int threads_all = omp_get_num_procs();
	cout << "Number of cpus in this machine: " << threads_all << endl;
	omp_set_num_threads(num_threads);
	cout << "Use " << num_threads << " threads" << endl;
	#endif
	double h = 1.0/(N+1.0);
	double res=0.0;
	double tol=1e-8;
	double rel_res=0.0;
	int iter=0;
	double res0=Residual(N,U,F);
	cout << "Initail residual is " << res0 << endl;
	rel_res=tol+1.0;
	while (rel_res>tol){
		#pragma omp parallel shared(U)
		{
			//red points
			#pragma omp for
			for (int j = 1; j <= N; j++) {
				int pt=-1;
				if (j%2 ==0){
					//even column
					pt=2;
				}else{
					//odd column
					pt=1;
				}
				for (int i = pt; i <= N; i+=2) {
					//rows first, in the inner loop since it's stored in row order.
					U[(N+2)*j+i] = 0.25 *
					(h * h * F[(N+2)*j+i] + U[(N+2)*j+i-1] + U[(N+2)*(j-1)+i]
					+ U[(N+2)*j+i+1]+ U[(N+2)*(j+1)+i]);
				}
			}
			//guarentee all red points is updated.
			#pragma omp barrier
			//black points
			#pragma omp for
			for (int j = 1; j <= N; j++) {
				int pt=-1;
				if (j%2 ==0){
					//even column
					pt=1;
				}else{
					//odd column
					pt=2;
				}
				for (int i = pt; i <= N; i+=2) {
					//rows first, in the inner loop since it's stored in row order.
					U[(N+2)*j+i] = 0.25 *
					(h * h * F[(N+2)*j+i] + U[(N+2)*j+i-1] + U[(N+2)*(j-1)+i]
					+ U[(N+2)*j+i+1]+ U[(N+2)*(j+1)+i]);
				}
			}
		}
		res=Residual(N,U,F);

		rel_res=res/res0;
		// if (iter%(maxit/10)==0){
		// 	std::cout << "Relative residual is " << rel_res << std::endl;
		// }
		iter++;
		if (iter>maxit){
			cout << "Max iteration reached: " << maxit <<endl;
			break;
		}
	}
}



__global__ void gs2D_gpu_kernel_black(int N, double h, double *U_new, double *U, double *F) {
  __shared__ double smem[BLOCK_DIM][BLOCK_DIM];
  smem[threadIdx.x][threadIdx.y]=0.0;
  if (blockIdx.x*BLOCK_DIM_IN+threadIdx.x<N+2 &&
    blockIdx.y*BLOCK_DIM_IN+threadIdx.y<N+2){
      smem[threadIdx.x][threadIdx.y]=U[(blockIdx.y*BLOCK_DIM_IN+threadIdx.y)*(N+2)+blockIdx.x*BLOCK_DIM_IN+threadIdx.x];
  }
  __syncthreads();
  if ((blockIdx.x*BLOCK_DIM_IN+threadIdx.x+blockIdx.y*BLOCK_DIM_IN+threadIdx.y)%2==0){
    if (threadIdx.x<=BLOCK_DIM_IN && threadIdx.x>=1 &&
      threadIdx.y<=BLOCK_DIM_IN && threadIdx.y>=1){
      if (blockIdx.x*BLOCK_DIM_IN+threadIdx.x<N+1 &&
        blockIdx.x*BLOCK_DIM_IN+threadIdx.x>0 &&
        blockIdx.y*BLOCK_DIM_IN+threadIdx.y<N+1 &&
        blockIdx.y*BLOCK_DIM_IN+threadIdx.y>0){
          U_new[(blockIdx.y*BLOCK_DIM_IN+threadIdx.y)*(N+2)+blockIdx.x*BLOCK_DIM_IN+threadIdx.x]=
          0.25 *
          (h * h * F[(blockIdx.y*BLOCK_DIM_IN+threadIdx.y)*(N+2)+blockIdx.x*BLOCK_DIM_IN+threadIdx.x] + smem[threadIdx.x-1][threadIdx.y] + smem[threadIdx.x+1][threadIdx.y]
          + smem[threadIdx.x][threadIdx.y-1]+ smem[threadIdx.x][threadIdx.y+1]);
      }
    }
  }
}

__global__ void gs2D_gpu_kernel_red(int N, double h, double *U_new, double *U, double *F) {
  __shared__ double smem[BLOCK_DIM][BLOCK_DIM];
  smem[threadIdx.x][threadIdx.y]=0.0;
  if (blockIdx.x*BLOCK_DIM_IN+threadIdx.x<N+2 &&
    blockIdx.y*BLOCK_DIM_IN+threadIdx.y<N+2){
      smem[threadIdx.x][threadIdx.y]=U[(blockIdx.y*BLOCK_DIM_IN+threadIdx.y)*(N+2)+blockIdx.x*BLOCK_DIM_IN+threadIdx.x];
  }
  __syncthreads();
  if ((blockIdx.x*BLOCK_DIM_IN+threadIdx.x+blockIdx.y*BLOCK_DIM_IN+threadIdx.y)%2==1){
    if (threadIdx.x<=BLOCK_DIM_IN && threadIdx.x>=1 &&
      threadIdx.y<=BLOCK_DIM_IN && threadIdx.y>=1){
      if (blockIdx.x*BLOCK_DIM_IN+threadIdx.x<N+1 &&
        blockIdx.x*BLOCK_DIM_IN+threadIdx.x>0 &&
        blockIdx.y*BLOCK_DIM_IN+threadIdx.y<N+1 &&
        blockIdx.y*BLOCK_DIM_IN+threadIdx.y>0){
          U_new[(blockIdx.y*BLOCK_DIM_IN+threadIdx.y)*(N+2)+blockIdx.x*BLOCK_DIM_IN+threadIdx.x]=
          0.25 *
          (h * h * F[(blockIdx.y*BLOCK_DIM_IN+threadIdx.y)*(N+2)+blockIdx.x*BLOCK_DIM_IN+threadIdx.x] + smem[threadIdx.x-1][threadIdx.y] + smem[threadIdx.x+1][threadIdx.y]
          + smem[threadIdx.x][threadIdx.y-1]+ smem[threadIdx.x][threadIdx.y+1]);
      }
    }
  }

}



void gs2D_gpu(int N, double *U, double *F, int maxit){
  double h = 1.0/(N+1.0);
  double res=0.0;
  double tol=1e-8;
  double rel_res=0.0;
  int iter=0;

  double *U_d, *F_d;
  cudaMalloc(&U_d, (N+2)*(N+2)*sizeof(double));
  cudaMalloc(&F_d, (N+2)*(N+2)*sizeof(double));
  cudaMemcpy(U_d, U, (N+2)*(N+2)*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(F_d, F, (N+2)*(N+2)*sizeof(double),cudaMemcpyHostToDevice);

  // cudaMalloc(&U_new, (N+2)*(N+2)*sizeof(double));
  // cudaMemcpy(U_new, U_d, (N+2)*(N+2)*sizeof(double),cudaMemcpyDeviceToDevice);
  // memset(U_new, 0, sizeof(double) * (N+2)*(N+2));
  // __shared__ double smem[BLOCK_DIM][BLOCK_DIM];
  double res0=Residual(N,U,F);
  cout << "Initail residual is " << res0 << endl;
  rel_res=tol+1.0;
  dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
  dim3 gridDim((N-1)/(BLOCK_DIM_IN)+1, (N-1)/(BLOCK_DIM_IN)+1);
  while (rel_res>tol){
      gs2D_gpu_kernel_black<<<gridDim,blockDim>>>(N, h, U_d, U_d, F_d);
      cudaDeviceSynchronize();
      // cudaMemcpy(U_d, U_new, (N+2)*(N+2)*sizeof(double),cudaMemcpyDeviceToDevice);
      gs2D_gpu_kernel_red<<<gridDim,blockDim>>>(N, h, U_d, U_d, F_d);
      cudaMemcpy(U,U_d,(N+2)*(N+2)*sizeof(double),cudaMemcpyDeviceToHost);
      res=Residual(N,U,F);
      rel_res=res/res0;
      // if (iter%(maxit/10)==0){
      // 	std::cout << "Relative residual is " << rel_res << std::endl;
      // }
      iter++;
      if (iter>maxit){
        cout << "Max iteration reached: " << maxit <<endl;
        cout << "Remaining res: " << rel_res <<endl;
        break;
      }
    }
    cout << "Remaining res: " << rel_res <<endl;
    // cudaFree(U_new);
  }

int main(int argc, char **argv) {

  cout << "Please input N(default=10): " << endl;
  int N = 10;
  cin >> N;
  cout << "Please input num of threads(default=1): " << endl;
  int num_threads = 1;
  cin >> num_threads;
  int maxit=10000;
  //allocate
  double *U = (double*) malloc ((N+2)*(N+2)*sizeof(double));
  double *F = (double*) malloc ((N+2)*(N+2)*sizeof(double));
  //initialize
  memset(U,0,(N+2)*(N+2)*sizeof(double));
  memset(F,0,(N+2)*(N+2)*sizeof(double));
  for (int i=0;i<(N+2)*(N+2);i++){
    F[i]=1.0;
  }
  Timer t;
  t.tic();
  gs2D_cpu(N, U, F, maxit,num_threads);
  printf("CPU Bandwidth = %f GB/s\n", maxit*10*(N+2)*(N+2)*sizeof(double) / (t.toc())/1e9);
  cout << "CPU Elapse time=" << t.toc() << "s" <<endl;

  memset(U,0,(N+2)*(N+2)*sizeof(double));
  t.tic();
  gs2D_gpu(N, U, F, maxit);
  printf("GPU Bandwidth = %f GB/s\n", maxit*10*(N+2)*(N+2)*sizeof(double) / (t.toc())/1e9);
  cout << "GPU Elapse time=" << t.toc() << "s" <<endl;

  free(U);
  free(F);
  return 0;
}
