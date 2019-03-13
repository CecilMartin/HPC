//Zhe Chen
#include <iostream>
#include <cmath>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "utils.h"

using namespace std;


double Residual(int N, double *U, double *F){
	double h=1.0/(N+1.0);
	double res=0.0, res_1=0.0;
	#pragma omp parallel for shared(U,F)\
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



void Gaiss_Seidel_Black_Red(int N, double *U, double *F, int maxit, int num_threads){
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
		if (iter%(maxit/10)==0){
			std::cout << "Relative residual is " << rel_res << std::endl;
		}
		iter++;
		if (iter>maxit){
			cout << "Max iteration reached: " << maxit <<endl;
			break;
		}
	}
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
	#pragma omp for
	for (int i=0; i<(N+2)*(N+2); i++){
		U[i]=0.0;
		F[i]=1.0;
	}
	Timer t;
	t.tic();
	Gaiss_Seidel_Black_Red(N, U, F, maxit,num_threads);
	cout << "Elapse time=" << t.toc() << "s" <<endl;
	free(U);
	free(F);
	return 0;
}
