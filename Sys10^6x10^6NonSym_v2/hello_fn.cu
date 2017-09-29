#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cusparse_v2.h"
#include "cublas_v2.h"
#include "hello.cuh"

#define imin(a,b) (a<b?a:b)
#define CLEANUP(s)      \
do{                     \
   printf("%s\n",s);    \
   if(I)     free(I);   \
   if(J)     free(J);   \
   if(val)   free(val); \
   if(r0)     free(r0);   \
   if(csrRowPtrA)  cudaFree(csrRowPtrA);   \
   if(csrColIndA)  cudaFree(csrColIndA);   \
   if(valA)  cudaFree(valA);   \
   if(x)    cudaFree(x);     \
   if(r)    cudaFree(r);     \
   if(d_Ax)   cudaFree(d_Ax);    \
   if(descrA)          cusparseDestroyMatDescr(descrA);  \
   if(cublasHandle)   cublasDestroy(cublasHandle);   \
   if(handle) cusparseDestroy(handle);   \
   cudaDeviceReset(); \
   fflush(stdout); \
} while(0)

/*
__global__ void set(double *dx,int N)
{
 int tid=threadIdx.x+blockIdx.x*blockDim.x ;
 if (tid<N)
 dx[tid]=0.0;
}
*/

extern "C"

{
 double *solverbicg(int* J,double* val,int* I,double* x0,double* r0, int N,int nz)
{

//const size_t sz=size_t(N)*sizeof(int);
/*
FILE *p1;
p1=fopen("x0.dat","w+");
FILE *p2;
p2=fopen("datosiniciales.dat","w+");
FILE *p3;
p3=fopen("csrRowPtrA.dat","w+");
FILE *p4;
p4=fopen("csrColIndA.dat","w+");
FILE *p5;
p5=fopen("d_valsILU0.dat","w+");
*/
FILE *p6;
p6=fopen("Ax.dat","w+");
//FILE *p7;
//p5=fopen("d_valsILU0.dat","w+");
//const int threadsPerBlock = 1024;
//const int blocksPerGrid = imin( 32, (N+threadsPerBlock-1) / threadsPerBlock );
int i,k;  
const float tol = 1e-16f;
const int maxit = 1;
double alpha=1.0,  beta=0.0,  alfa=0.0,alphan=0.0, rho=0.0, rhop=0.0, temp=0,temp2=0, nrmr=0, nrmr0=0, uno=1.0, cero=0.0, omega=1.0, omegan=0.0;
double *d_Ax=0, *test=0,*test1=0, *valA=0,*x=0, *r=0, *d_valsILU0=0, *valL=0, *valU=0, *f=0, *p=0, *rw=0, *t=0, *ph=0, *q=0, *s=0;
int *csrRowPtrA=0, *csrColIndA=0, *csrRowPtrL=0, *csrRowPtrU=0, *csrColIndL=0, *csrColIndU=0;
// *test2=0, *test3=0;
int nzILU0 = 2*N-1;
cudaError_t cudaStat1,cudaStat2,cudaStat3, cudaStat4,cudaStat5,cudaStat6,cudaStat7, cudaStat8, cudaStat9, cudaStat10,cudaStat11, cudaStat12, cudaStat13;
cublasStatus_t cublasStatus1;
cublasHandle_t cublasHandle=0;
cusparseStatus_t cusparseStatus1;
cusparseHandle_t handle;
cusparseMatDescr_t descrA;

cusparseStatus1=cusparseCreate(&handle);
 if(cusparseStatus1!=CUSPARSE_STATUS_SUCCESS){
   CLEANUP("Cusparse create handle failed\n");
 }

cublasStatus1=cublasCreate(&cublasHandle);
  if(cublasStatus1!=CUBLAS_STATUS_SUCCESS){
     CLEANUP("Cublas create handle failed \n");}


cusparseStatus1=cusparseCreateMatDescr(&descrA);
  if(cusparseStatus1!=CUSPARSE_STATUS_SUCCESS){
        printf("Descriptor creation failed\n");
 }//Set matrix type and index base

cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);


test = (double *)malloc(sizeof(double)*nz);
test1 = (double *)malloc(sizeof(double)*N);
//test2 = (int *)malloc(sizeof(int )*N);
//test3 = (int *)malloc(sizeof(int )*nz);
if((!test)||(!test1)){CLEANUP("Memory on host failed,test\n");}

/*
   for(i=0;i<nz;i++){
       fprintf(p2,"i=%d,J=%d,val=%7.3f,I=%d, r=%7.3f \n",i,J[i],val[i],I[i],r0[i]);}
   printf("N=%d  nz=%d\n",N,nz);
*/


    cudaStat1=cudaMalloc((void **)&csrRowPtrA, (N+1)*sizeof(int));
    cudaStat2=cudaMalloc((void **)&csrColIndA, nz*sizeof(int));
    cudaStat3=cudaMalloc((void **)&valA, nz*sizeof(double));
    cudaStat4=cudaMalloc((void **)&x, N*sizeof(double));
    cudaStat5=cudaMalloc((void **)&r, N*sizeof(double));
    cudaStat7=cudaMalloc((void **)&d_Ax, N*sizeof(double));
    cudaStat8=cudaMalloc((void **)&csrRowPtrL, (N+1)*sizeof(int));
    cudaStat9=cudaMalloc((void **)&csrRowPtrU, (N+1)*sizeof(int));
    cudaStat10=cudaMalloc((void **)&csrColIndL, nz*sizeof(int));
    cudaStat11=cudaMalloc((void **)&csrColIndU, nz*sizeof(int));
    cudaStat12=cudaMalloc((void **)&f, N*sizeof(double));
    if((cudaStat1!=cudaSuccess)||(cudaStat2!=cudaSuccess)||(cudaStat3!=cudaSuccess)||(cudaStat4!=cudaSuccess)||(cudaStat5!=cudaSuccess)||(cudaStat7!=cudaSuccess)||(cudaStat8!=cudaSuccess)||(cudaStat9!=cudaSuccess)||(cudaStat10!=cudaSuccess)||(cudaStat11!=cudaSuccess)||(cudaStat12!=cudaSuccess)){printf("allocate memory on device"); }
    
//set initial values
    cudaStat1=cudaMemcpy(csrColIndA, J, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaStat2=cudaMemcpy(csrRowPtrA, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaStat3=cudaMemcpy(valA, val, nz*sizeof(double), cudaMemcpyHostToDevice);
    cudaStat4=cudaMemcpy(x, x0, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaStat5=cudaMemcpy(f, r0, N*sizeof(double), cudaMemcpyHostToDevice);
    if((cudaStat1!=cudaSuccess)||(cudaStat2!=cudaSuccess)||(cudaStat3!=cudaSuccess)||(cudaStat5!=cudaSuccess)){printf("Memcpy from Host to Device failed\n");
          } 

//   set<<<blocksPerGrid,threadsPerBlock>>>(x,N);
  

//  cudaStat1=cudaMemset(x,0,sz);
//if(cudaStat1!=cudaSuccess){printf("set x0 to 0 failed\n");
//          }

/*
   cudaStat1=cudaMemcpy(test2,x,(N)*sizeof(int),cudaMemcpyDeviceToHost);
   if(cudaStat1!=cudaSuccess){printf("Memcpy from Host to Device failed\n");
          }
printf("x de la copia \n");

   for(i=0;i<N;i++)
    {
      fprintf(p1,"%d\n",test2[i]);
    }
*/


 // Preconditioned Conjugate Gradient using ILU.
 
    cudaStat1=cudaMalloc((void **)&d_valsILU0, nz*sizeof(double));
    cudaStat2=cudaMalloc((void **)&q, N*sizeof(double));
    cudaStat7=cudaMalloc((void **)&valL, nz*sizeof(double));
    cudaStat8=cudaMalloc((void **)&valU, nz*sizeof(double));
    cudaStat9=cudaMalloc((void **)&p, N*sizeof(double));
    cudaStat10=cudaMalloc((void **)&rw, N*sizeof(double));
    cudaStat11=cudaMalloc((void **)&t, N*sizeof(double));
    cudaStat12=cudaMalloc((void **)&ph, N*sizeof(double));
    cudaStat13=cudaMalloc((void **)&s, N*sizeof(double));
    if((cudaStat1!=cudaSuccess)||(cudaStat2!=cudaSuccess)||(cudaStat7!=cudaSuccess)||(cudaStat8!=cudaSuccess)||(cudaStat9!=cudaSuccess)||(cudaStat10!=cudaSuccess)||(cudaStat11!=cudaSuccess)||(cudaStat12!=cudaSuccess)||(cudaStat13!=cudaSuccess)){printf("allocate memory on device 2"); }


/* create the analysis info object for the A matrix */
    cusparseSolveAnalysisInfo_t infoA = 0;
    cusparseCreateSolveAnalysisInfo(&infoA);

/* Perform the analysis for the Non-Transpose case */
  cusparseStatus1 =cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             N, nz, descrA, valA,  csrRowPtrA, csrColIndA, infoA);
    if((cusparseStatus1!=CUSPARSE_STATUS_SUCCESS)){printf("Dcsrsv_analysis (1)failed\n");}

 /* Copy A data to ILU0 vals as input*/
   cudaStat1=cudaMemcpy(d_valsILU0, valA, nz*sizeof(double), cudaMemcpyDeviceToDevice);
   if((cudaStat1!=cudaSuccess)){printf("cudaMemcpyDeviceToDevice (2) failed \n");}

    /* generate the Incomplete LU factor H for the matrix A using cudsparseScsrilu0 */
  cusparseStatus1=cusparseDcsrilu0(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, descrA, d_valsILU0,csrRowPtrA, csrColIndA,  infoA);if(cusparseStatus1!=CUSPARSE_STATUS_SUCCESS){printf("cudaMemcpyDeviceToDevice (3) failed\n");}
/*
cudaStat1=cudaMemcpy(test2,csrRowPtrA, (N+1)*sizeof(int), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP(" d_row cudaMemcpyDeviceToHost failed\n");}
    printf(" d_row \n ");
    for(i=0;i<(N+1);i++){
     fprintf(p3,"%d\t",test2[i]);}
     printf("\n");

cudaStat1=cudaMemcpy(test3,csrColIndA, nz*sizeof(int), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP(" d_row cudaMemcpyDeviceToHost failed\n");}
    printf(" d_col \n ");
    for(i=0;i<nz;i++){
     fprintf(p4,"%d\t",test3[i]);}
     printf("\n");

cudaStat1=cudaMemcpy(test,d_valsILU0, nz*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP(" d_valsILU0 cudaMemcpyDeviceToHost failed\n");}
    printf(" d_valsILU0 \n ");
    for(i=0;i<nz;i++){
     fprintf(p5,"%e\t",test[i]);}
     printf("\n");

*/


   /* Copy ILU0 data to valL and valU*/
    cudaStat1=cudaMemcpy(valL, d_valsILU0, nz*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaStat2=cudaMemcpy(valU, d_valsILU0, nz*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaStat3=cudaMemcpy(csrRowPtrL,csrRowPtrA, (N+1)*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaStat4=cudaMemcpy(csrRowPtrU,csrRowPtrA, (N+1)*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaStat5=cudaMemcpy(csrColIndL,csrColIndA, nz*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaStat6=cudaMemcpy(csrColIndU,csrColIndA, nz*sizeof(int), cudaMemcpyDeviceToDevice);
    if((cudaStat1!=cudaSuccess)||(cudaStat2!=cudaSuccess)||(cudaStat3!=cudaSuccess)||(cudaStat4!=cudaSuccess)||(cudaStat5!=cudaSuccess)||(cudaStat6!=cudaSuccess)){printf("cudaMemcpyDeviceToDevice (4) failed\n");}

/*
cudaStat1=cudaMemcpy(test2,csrRowPtrL, (N+1)*sizeof(int), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP(" csrRowPtrL cudaMemcpyDeviceToHost  failed\n");}
    printf("  csrRowPtrL \n ");
    for(i=0;i<(N+1);i++){
     printf("%d\t",test2[i]);}
     printf("\n");

cudaStat1=cudaMemcpy(test2,csrRowPtrU, (N+1)*sizeof(int), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP(" csrRowPtrU  failed\n");}
    printf("csrRowPtrU  \n ");
    for(i=0;i<(N+1);i++){
     printf("%d\t",test2[i]);}
     printf("\n");

cudaStat1=cudaMemcpy(test3,csrColIndL, nz*sizeof(int), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP(" csrColIndL cudaMemcpyDeviceToHost failed\n");}
    printf(" csrColIndL \n ");
    for(i=0;i<nz;i++){
     printf("%d\t",test3[i]);}
     printf("\n");

cudaStat1=cudaMemcpy(test3,csrColIndU, nz*sizeof(int), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP(" csrColIndU cudaMemcpyDeviceToHost failed\n");}
    printf(" csrColIndU \n ");
    for(i=0;i<nz;i++){
     printf("%d\t",test3[i]);}
     printf("\n");


cudaStat1=cudaMemcpy(test,valL, nz*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP(" valL cudaMemcpyDeviceToHost failed\n");}
    printf(" valL \n ");
    for(i=0;i<nz;i++){
     printf("%e\t",test[i]);}
     printf("\n");

cudaStat1=cudaMemcpy(test,valU, nz*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP(" valU cudaMemcpyDeviceToHost failed\n");}
    printf(" valU \n ");
    for(i=0;i<nz;i++){
     printf("%e\t",test[i]);}
     printf("\n");
*/

   /* Create info objects for the ILU0 preconditioner */
    cusparseSolveAnalysisInfo_t infoU;
    cusparseCreateSolveAnalysisInfo(&infoU);
    cusparseSolveAnalysisInfo_t infoL;
    cusparseCreateSolveAnalysisInfo(&infoL);

    cusparseMatDescr_t descrL = 0;
    cusparseStatus1=cusparseCreateMatDescr(&descrL);
     if((cusparseStatus1!=CUSPARSE_STATUS_SUCCESS)){printf("cusparseCreateMatDescr(&descrL) (5) failed\n");}
    cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrL,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT);
    cusparseStatus1=cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, descrL, valL, csrRowPtrL, csrColIndL, infoL); 
    if(cusparseStatus1!=CUSPARSE_STATUS_SUCCESS){printf("cusparseDcsrsv_analysis (6) failed\n");}
    
    cusparseMatDescr_t descrU = 0;
    cusparseStatus1=cusparseCreateMatDescr(&descrU);
    if(cusparseStatus1!=CUSPARSE_STATUS_SUCCESS){printf("cusparseCreateMatDescr(&descrU) (7) failed\n");}
    cusparseSetMatType(descrU,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrU,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseStatus1=cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, descrU, valU, csrRowPtrU, csrColIndU, infoU);
    if((cusparseStatus1!=CUSPARSE_STATUS_SUCCESS)){printf("cusparseDcsrsv_analysis (8) failed\n");}
/*
    cudaStat1=cudaMemcpy(test,valA, nz*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP(" valU cudaMemcpyDeviceToHost failed\n");}
    printf(" valA \n ");
    for(i=0;i<nz;i++){
     printf("%e\t",test[i]);}
     printf("\n");

      cudaStat1=cudaMemcpy(test2,csrRowPtrA, (N+1)*sizeof(int), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP(" csrRowPtrU  failed\n");}
    printf("csrRowPtrA  \n ");
    for(i=0;i<(N+1);i++){
     printf("%d\t",test2[i]);}
     printf("\n");

     cudaStat1=cudaMemcpy(test3,csrColIndA, nz*sizeof(int), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP(" csrColIndU cudaMemcpyDeviceToHost failed\n");}
    printf(" csrColIndA \n ");
    for(i=0;i<nz;i++){
     printf("%d\t",test3[i]);}
     printf("\n");
    
     printf("alfa=%f \n",alpha);
     printf("beta=%f \n",beta);

  */   
   
    //1:compute initial Ax0 con x0=0     
    cusparseStatus1=cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N,N,nz,&alpha,descrA,valA,csrRowPtrA,csrColIndA,x,&beta,r);
    if(cusparseStatus1!=CUSPARSE_STATUS_SUCCESS){printf("cusparseDcsrmv (9) failed\n");}
/*
    cudaStat1=cudaMemcpy(test,r, N*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP("Memcpy from Device to Host failed(9)\n");}
    printf(" residuo Ax0, \n ");
    for(i=0;i<N;i++){
     printf("%e\t",test[i]);}
     printf("\n");   
*/
//we can avoid this step as r=Ax0=0 as x0=0 
//cublasDscal(n,-1.0,r,1);
cublasDaxpy(cublasHandle,N,&alpha,f,1,r,1);
/*
 cudaStat1=cudaMemcpy(test,r, N*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP("Memcpy from Device to Host failed(9)\n");}
    printf(" r=f=b=vect=b-Ax0, con Ax0=0, \n ");
    for(i=0;i<N;i++){
     printf("%e\t",test[i]);}
     printf("\n");
*/
//2: Set p=r and \tilde{r}=r

cublasDcopy(cublasHandle, N,r,1,p,1);
/*
cudaStat1=cudaMemcpy(test,p, N*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP("Memcpy from Device to Host failed(9)\n");}
    printf(" p=r, \n ");
    for(i=0;i<N;i++){
     printf("%e\t",test[i]);}
     printf("\n");
*/
cublasDcopy(cublasHandle, N,r,1,rw,1);
/*
cudaStat1=cudaMemcpy(test,rw, N*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP("Memcpy from Device to Host failed(9)\n");}
    printf(" rw=p, \n ");
    for(i=0;i<N;i++){
     printf("%e\t",test[i]);}
     printf("\n");
*/

   cublasDdot(cublasHandle,N,r,1,r,1,&rho);
   printf("rho=%f \n",rho);
   nrmr0=rho;

   k=1;

//3: repeat until convergence (based on max. it. and relative residual)
     for(i=0;i<maxit;i++)
   {
      
       //4: \rho=\tilde{r}^{T} r
       rhop=rho;
       cublasDdot(cublasHandle,N,rw,1,r,1,&rho);
     //  printf("rho=%f \n",rho);
       
       if (i>0)
       {
       //12:
       beta=(rho/rhop)*(alpha/omega);
  //     printf("beta=%f \n",beta); 
  
       //13: p=r+\beta (p- \omega v)
       
       cublasDaxpy(cublasHandle,N,&omegan,q,1,p,1);
/*
      cudaStat1=cudaMemcpy(test,p, N*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP("p-omega*v failed\n");}
    printf(" p-omega*v \n ");
    for(i=0;i<N;i++){
     printf("%e\t",test[i]);}
     printf("\n");
*/

       cublasDscal(cublasHandle,N,&beta,p,1);
/*
        cudaStat1=cudaMemcpy(test,p, N*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP("beta(p-omega*v) failed \n");}
    printf("beta( p-omega*v) \n ");
    for(i=0;i<N;i++){
     printf("%e\t",test[i]);}
     printf("\n");
*/
       cublasDaxpy(cublasHandle,N,&uno,r,1,p,1);
  /*      
     cudaStat1=cudaMemcpy(test,p, N*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP("r+beta(p-omega*v) failed \n");}
    printf("r + beta( p-omega*v) \n ");
    for(i=0;i<N;i++){
     printf("%e\t",test[i]);}
     printf("\n");
*/

        }


      //15: M\hat{p}=p sparse lower and upper triangular solves

cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N,&alpha, descrL, valL, csrRowPtrL, csrColIndL, infoL,p,t);
/*
cudaStat1=cudaMemcpy(test,t, N*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP("d_ y solve triangular system\n");}
    printf(" t \n ");
    for(i=0;i<N;i++){
     printf("%e\t",test[i]);}
     printf("\n");
*/

   cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N,&alpha, descrU, valU, csrRowPtrU, csrColIndU, infoU,t,ph);
/*
   cudaStat1=cudaMemcpy(test,ph, N*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP("ph \n");}
    printf(" ph   \n ");
    for(i=0;i<N;i++){
     printf("%e\t",test[i]);}
     printf("\n");
*/

//16
     cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N,N,nzILU0, &alpha,descrA,valA,csrRowPtrA,csrColIndA,ph,&beta,q);
/*
cudaStat1=cudaMemcpy(test,q, N*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP(" q=Aph \n");}
    printf(" q=Ap   \n ");
    for(i=0;i<N;i++){
     printf("%e\t",test[i]);}
     printf("\n");
//17:\alpha=\rho_{i} / (\tilde{r}^{T} q)
*/
   cublasDdot(cublasHandle,N,rw,1,q,1,&temp);
  //     printf("temp=rw*p=%f \n",temp);

   alfa=rho/temp;
// printf("alpha=%f \n",alfa);
   alphan=-alfa;
//18: s=r -\alpha q
   cublasDaxpy(cublasHandle,N,&alphan, q,1,r,1);
/*
  cudaStat1=cudaMemcpy(test,r,N*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP(" s=r -alpha q failed\n");}
    printf(" s=r -alpha  \n ");
    for(i=0;i<N;i++){
     printf("%e\t",test[i]);}
     printf("\n");
*/
    //19 x=x+\alpha \hat{p}

   cublasDaxpy(cublasHandle,N,&alfa, ph,1,x,1);
/*
   cudaStat1=cudaMemcpy(test,x,N*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP(" x failed\n");}
    printf("x \n ");
    for(i=0;i<N;i++){
     printf("%e\t",test[i]);}
     printf("\n");
*/
//20: check for convergence
    cublasDnrm2(cublasHandle,N,r,1, &nrmr);
    printf(" nrmr= %f \n", nrmr);
     printf("  nrmr/nrmr0 %f \n", nrmr/nrmr0);
     if(nrmr/nrmr0<tol)
        {break;}
   
    
//23:  M\hat{s}=p sparse lower and upper triangular solves
     cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N,&uno, descrL, valL, csrRowPtrL, csrColIndL, infoL,r,t);
  /*   
     cudaStat1=cudaMemcpy(test,t,N*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP(" t failed\n");}
    printf("t M*hat{s}=r \n ");
    for(i=0;i<N;i++){
     printf("%e\t",test[i]);}
     printf("\n");
*/

     cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_TRANSPOSE, N,&uno, descrU, valU, csrRowPtrU, csrColIndU, infoU,t,s);

/*
    cudaStat1=cudaMemcpy(test,s,N*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP("s failed\n");}
    printf("s M*hat{s}=r \n ");
    for(i=0;i<N;i++){
     printf("%e\t",test[i]);}
     printf("\n");
*/
    //24:t=A\hat{s} (sparse matrix-vector multiplication)

    cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,N,N,nzILU0,&uno,descrA, valA, csrRowPtrA,csrColIndA,s,&cero,t);
/*
    cudaStat1=cudaMemcpy(test,t,N*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP("t failed\n");}
    printf("t=A*hat{s} \n ");
    for(i=0;i<N;i++){
     printf(" %e\t",test[i]);}
     printf("\n");
  */
     //25 \omega=
    cublasDdot(cublasHandle,N,t,1,r,1,&temp);
    printf("temp=%f \n",temp);

    cublasDdot(cublasHandle,N,t,1,t,1,&temp2);
    printf("temp2=%f \n",temp2);


    omega=temp/temp2;
   // printf("omega=%f \n",omega); 

    //26:
    cublasDaxpy(cublasHandle,N,&omega,s,1,x,1);
/*
    cudaStat1=cudaMemcpy(test,x,N*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP("x_i failed\n");}
    printf("x_i \n ");
    for(i=0;i<N;i++){
     printf(" %e\t",test[i]);}
     printf("\n");
*/
    omegan=-omega;

    cublasDaxpy(cublasHandle,N,&omegan,t,1,r,1);
/*
    cudaStat1=cudaMemcpy(test,r,N*sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaStat1!=cudaSuccess){
           CLEANUP("r_i failed\n");}
    printf("r_i \n ");
    for(i=0;i<N;i++){
     printf(" %e\t",test[i]);}
     printf("\n");
*/
   
//20: check for convergence
    cublasDnrm2(cublasHandle,N,r,1, &nrmr);
    printf(" nrmr= %f \n", nrmr);
     printf("  nrmr/nrmr0 %f \n", nrmr/nrmr0);
     if(nrmr/nrmr0<tol)
        {break;}

   cudaThreadSynchronize();

    k++;


   }

   
     printf("residuo=%e, iteraciones=%d \n",nrmr/nrmr0,k);


    alpha=1.0;
    beta=0.0;

 cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,N,N,nzILU0,&uno,descrA, valA, csrRowPtrA,csrColIndA,s,&cero,t);


cusparseStatus1=cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descrA, valA, csrRowPtrA,csrColIndA, x, &beta, d_Ax);
    if(cusparseStatus1!=CUSPARSE_STATUS_SUCCESS){
    CLEANUP("Ax0 performing failed\n");}


  cudaStat1=cudaMemcpy(test, d_Ax, N*sizeof(double), cudaMemcpyDeviceToHost);
  if(cudaStat1!=cudaSuccess){
    CLEANUP("Memcpy from Device to Host failed\n"); }
  printf(" d_Ax_(j+1), \n ");
  for(i=0;i<N;i++){
     fprintf(p6,"%e\n",test[i]);
    // test3[i]=test2[i];
  }
  printf("\n");











//printf("iteration = %3d, residual = %e\n", k, r1);

  //  fclose(p1);
  //  fclose(p2);
  //  fclose(p3);
  //  fclose(p4);
    fclose(p6);
    /* Destroy paramters */
    cusparseDestroySolveAnalysisInfo(infoA);
    cusparseDestroySolveAnalysisInfo(infoU);
    cusparseDestroySolveAnalysisInfo(infoL);

    /* Destroy contexts */

    cusparseDestroy(handle);
    cublasDestroy(cublasHandle);

    free(I);
    free(J);
    free(val);
    free(r0);
    free(test);
    cudaFree(csrRowPtrA);
    cudaFree(csrColIndA);
    cudaFree(csrRowPtrL);
    cudaFree(csrColIndL);
    cudaFree(csrRowPtrU);
    cudaFree(csrColIndU);
    cudaFree(valA);
    cudaFree(x);
    cudaFree(r);
    cudaFree(d_Ax);
    cudaFree(valL);
    cudaFree(valU);
    cudaFree(t);
    cudaFree(ph);


    cudaDeviceReset();

 return test1;

}
}
