#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cusparse_v2.h"
#include "cublas_v2.h"
#include "hello.cuh"

main(int argc, char **argv)
{

int i, N=1000000, nz = 2999998, *I = 0, *J =0;
double   *x=0,*r=0, *val = 0, *answer=0;


FILE *fp;
fp=fopen("ColInd.dat","r+");
FILE *fp1;
fp1=fopen("Vals.dat","r+");
FILE *fp2;
fp2=fopen("RowPtr.dat","r+");
FILE *fp3;
fp3=fopen("x0.dat","r+");
FILE *fp4;
fp4=fopen("Vect.dat","r+");

answer = (double *)malloc(sizeof(double)*N);
J = (int *)malloc(sizeof(int)*nz); 
val = (double *)malloc(sizeof(double)*nz);
I = (int *)malloc(sizeof(int)*(N+1)); //vector de posicion de fila de csr
x = (double *)malloc(sizeof(double)*N);
r = (double *)malloc(sizeof(double)*N);
  if((!I)||(!J)||(!val)||(!r)){
  printf("malloc failed\n"); return 1;}

for(i=0;i<nz;i++){
      fscanf(fp,"%d",&J[i]);}
for(i=0;i<nz;i++){
      fscanf(fp1,"%lf",&val[i]);}
for(i=0;i<N+1;i++){
      fscanf(fp2,"%d",&I[i]);}
for(i=0;i<N;i++){
      fscanf(fp3,"%lf",&x[i]);}
for(i=0;i<N;i++){
      fscanf(fp4,"%lf",&r[i]);}

//printf("alpha=%7.2f, alpham1=%7.2f, beta=%7.2f \n ",alpha,alpham1,beta);
/*
for(i=0;i<nz;i++){
 printf("i=%d,J=%d,val=%7.3f,I=%d, r=%7.3f \n",i,J[i],val[i],I[i],r[i]);}
*/


answer=solverbicg((int*) J,(double*) val,(int*) I,(double*) x,(double *) r, N, nz);
/*
printf("Desde main\n");
for(i=0;i<N;i++)
{
printf("answer%d=%f \n",i,*(answer+i));
}
*/


return 0;


}
