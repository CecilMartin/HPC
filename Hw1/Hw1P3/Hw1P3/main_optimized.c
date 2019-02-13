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



void MMult0( int n,
            double *a,
            double *b) {
    double h=1.0/(1+n);
    for (int i = 0; i < n; i++) {
        if(i==0){
            b[i]=(2*a[i]-a[i+1])/h/h;
        }
        else if(i==n-1){
            b[i]=(2*a[i]-a[i-1])/h/h;
        }
        else{
            b[i]=(2*a[i]-a[i-1]-a[i+1])/h/h;
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

int Jacobi(double *x, int N, double *b,int maxit, double err_limit){
    int iter,i;
    double err0,err=0.0,h=1.0/(N+1);
    double *b0, *x_new;
    b0=(double *)malloc(sizeof(double)*N);
    x_new=(double *)malloc(sizeof(double)*N);
    for(i=0;i<N;i++){
        b0[i]=0.0;
        x_new[i]=0.0;
    }
    MMult0(N, x, b0);
    err0=diff_vect(b0, b, N);
    printf("err=%f\n",err0);
    for(iter=0;iter<maxit;iter++){
        for(i=0;i<N;i++){
            if(i==0){
                x_new[i]=(b[i]*h*h+x[i+1])/2;
            }
            else if(i==N-1){
                x_new[i]=(b[i]*h*h+x[i-1])/2;
            }
            else{
                x_new[i]=(b[i]*h*h+x[i+1]+x[i-1])/2;
            }
        }
        MMult0(N, x_new, b0);
        err=diff_vect(b0, b, N);
        if (iter%(maxit/10)==0) {
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

int GS(double *x, int N, double *b,int maxit, double err_limit){
    int iter,i;
    double err0,err=0.0,h=1.0/(N+1);
    double *b0;
    b0=(double *)malloc(sizeof(double)*N);
    for(i=0;i<N;i++){
        b0[i]=0.0;
    }
    MMult0(N, x, b0);
    err0=diff_vect(b0, b, N);
    printf("err=%f\n",err0);
    for(iter=0;iter<maxit;iter++){
        for(i=0;i<N;i++){
            if(i==0){
                x[i]=(b[i]*h*h+x[i+1])/2;
            }
            else if(i==N-1){
                x[i]=(b[i]*h*h+x[i-1])/2;
            }
            else{
                x[i]=(b[i]*h*h+x[i+1]+x[i-1])/2;
            }
        }
        MMult0(N, x, b0);
        err=diff_vect(b0, b, N);
//        for(i=1;i<N;i++){
//            printf("%f  ",x[i]);
//        }
//        printf("\n");
        if (iter%(maxit/10)==0) {
            printf("iter=%d,err=%f\n",iter,err);
        }
        //printf("err=%f\n",err);
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
    int n=10000, maxit=100, flag;
    double *x=0,*b=0,h=1.0/(1+n), err_limit=1.0e-6,tim;
    int i;
    if(n<3){
        printf("ERROR!!!!!!!!!!!!!!!!!!!");
    }
    x=(double *)malloc(sizeof(double)*n);
    b=(double *)malloc(sizeof(double)*n);
    for(i=0;i<n;i++){
        x[i]=0.0;
        b[i]=1.0;
    }
    start = clock();
    flag=GS(x, n, b, maxit,err_limit);
    finish= clock();
    tim = (double)(finish- start) / CLOCKS_PER_SEC;
    printf("Total time is %f.\n",tim);
}
