//
//  main.c
//  Hw1P3
//
//  Created by Zhe Chen on 2/11/19.
//  Copyright © 2019 Zhe Chen. All rights reserved.
//
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>



void MMult0( int m, int k, double **a,
            double *b,
            double *c) {
    for (int i = 0; i < m; i++) {
        c[i]=0.0;
        for (int p = 0; p < k; p++) {
            c[i]+=a[i][p]*b[p];
        }
    }
}

double norm2(double *x,int n){
    int i;
    double sum=0;
    for(i=0;i<n;i++){
        sum=sum+pow(x[i],2);
    }
    return sqrt(sum);
}

double diff_vect(double *a, double *b, int n){
    int i=0;
    double err=0.0;
    for(i=0;i<n;i++){
        err=err+pow((a[i]-b[i]),2);
        //printf("%f\n",err);
    }
    return sqrt(err);
}

int Jacobi(double *x, int N,double **a, double *b,int maxit, double err_limit){
    int iter,i,j;
    double err0,temp,err=0.0;
    double *b0, *x_new;
    b0=(double *)malloc(sizeof(double)*N);
    x_new=(double *)malloc(sizeof(double)*N);
    for(i=0;i<N;i++){
        b0[i]=0.0;
        x_new[i]=0.0;
    }
    MMult0(N, N, a, x, b0);
    err0=diff_vect(b0, b, N);
    printf("err=%f\n",err0);
    for(iter=0;iter<maxit;iter++){
        for(i=0;i<N;i++){
            temp=0.0;
            for(j=0;j<N;j++){
                if(i!=j){
                    temp=temp+a[i][j]*x[j];
                }
            }
            //x[i]=(b[i]-temp)/(a[i][i]);
            x_new[i]=(b[i]-temp)/(a[i][i]);
        }
        MMult0(N, N, a, x_new, b0);
        err=diff_vect(b0, b, N);
        if (iter%100==0) {
            printf("iter=%d,err=%f\n",iter,err);
        }
        //printf("err=%f\n",err);
        memcpy(x, x_new, sizeof(double)*N);
        if(err/err0<err_limit){
            printf("err=%f, iter=%d, flag=%d\n",err,iter+1,-1);
            return -1;
        }

    }
    printf("err=%f, iter=%d, flag=%d\n",err,iter,0);
    return 0;

}
int GS(double *x, int N,double **a, double *b,int maxit, double err_limit){
    int iter,i,j;
    double err0,temp,err=0.0;
    double *b0;
    b0=(double *)malloc(sizeof(double)*N);
    //x_new=(double *)malloc(sizeof(double)*N);
    for(i=0;i<N;i++){
        b0[i]=0.0;
        //x_new[i]=0.0;
    }
    MMult0(N, N, a, x, b0);
    err0=diff_vect(b0, b, N);
    printf("err=%f\n",err0);
    for(iter=0;iter<maxit;iter++){
        for(i=0;i<N;i++){
            temp=0.0;
            for(j=0;j<N;j++){
                if(j!=i){
                    temp=temp+a[i][j]*x[j];
                }
            }
            x[i]=(b[i]-temp)/(a[i][i]);
            //x_new[i]=(b[i]-temp)/(a[i][i]);
        }
        MMult0(N, N, a, x, b0);
        err=diff_vect(b0, b, N);
        if (iter%100==0) {
            printf("iter=%d,err=%f\n",iter,err);
        }
        //printf("err=%f\n",err);
        //memcpy(x, x_new, sizeof(double)*N);
        if(err/err0<err_limit){
            printf("err=%f, iter=%d, flag=%d\n",err,iter+1,-1);
            return -1;
        }
    }
    printf("err=%f, iter=%d, flag=%d\n",err,iter,0);
    return 0;

}

int main(){
    clock_t start, finish;
    int n=100, maxit=50000, flag;
    double *a[n],*x=0,*b=0,h=1.0/(1+n), err_limit=1.0e-6,tim;
    int i,j;
    x=(double *)malloc(sizeof(double)*n);
    b=(double *)malloc(sizeof(double)*n);
    for(i=0;i<n;i++){
        x[i]=0.0;
        b[i]=1.0;
        double *temp=(double *)malloc(sizeof(double)*n);
        for(j=0;j<n;j++){
            if(i==j){
                temp[j]=2/h/h;
            }
            else if(j==i-1 || j==i+1){
                temp[j]=-1/h/h;
            }
            else{
                temp[j]=0.0;
            }
        }
        //memcpy(a[i],&temp,sizeof(double)*n);
        a[i]=temp;
    }
    start = clock();
    flag=Jacobi(x, n, a, b, maxit,err_limit);
    finish= clock();
    tim = (double)(finish- start) / CLOCKS_PER_SEC;
    printf("Total time is %f.\n",tim);
}
