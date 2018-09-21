#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
   
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

#ifdef GPU

#include <math.h>

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}
int* completion(int TA, int TB,char *h_A,int m,int k,char *h_B,int n,char* bias){
    int completion_A = (((4 - k%4) == 4)? 0:(4 - k%4))+k;
    char *h_A_new;
    cudaError_t status = cudaMalloc((void **)&h_A_new, completion_A*m*sizeof(char));
    check_error(status);

	int completion_B_row;
	int completion_B_col;
    completion_B_row = (((4 - k%4) == 4)?0:(4 - k%4)) + k;
    completion_B_col = (((4 - n%4) == 4)?(0):(4 - n%4)) + n;
   
    
    char *h_B_new;
    status = cudaMalloc((void **)&h_B_new, completion_B_row*completion_B_col*sizeof(char));
    check_error(status);


    int *h_C_new;
    status = cudaMalloc((void **)&h_C_new, m*completion_B_col*sizeof(int));
    check_error(status);

    int *h_C;
    status = cudaMalloc((void **)&h_C, m*n*sizeof(int));
    check_error(status);
    addPaddingA_gpu(h_A_new,h_A,m,completion_A,m,k,TA);
   
   
   
    addPaddingB_gpu(h_B_new,h_B,completion_B_row,completion_B_col,k,n,TB);
	int a = 1;int b = 0;
   
   
   
   
   
    cublasHandle_t handle = blas_handle();
   
   
	cublasStatus_t status1 = cublasGemmEx(
		handle,		
		CUBLAS_OP_N,  
		CUBLAS_OP_N,  
		completion_B_col,         
		m,         
		completion_B_row,         
		&a,            
		h_B_new,           
		CUDA_R_8I,
		completion_B_col,         
		h_A_new,           
		CUDA_R_8I,
	    completion_A,         
		&b,            
		h_C_new,           
		CUDA_R_32I,
		completion_B_col,          
		CUDA_R_32I,
		CUBLAS_GEMM_ALGO1
	);
    check_error(status1);
    rmPadding_gpu(h_C,h_C_new,m,completion_B_col,m,n);
    if (bias!=0)
        add_bias_gpu_fc(h_C,m*completion_B_col,bias);

    cuda_free((float*)h_A_new);
    cuda_free((float*)h_B_new);
    cuda_free((float*)h_C_new);
    return h_C;
}
void int_gemm_ongpu(int TA, int TB, int M, int N, int K, int ALPHA, 
        char *A_gpu, int lda, 
        char *B_gpu, int ldb,
        int BETA,
        char *C_gpu, int ldc, char shift_bit)
{
    int *C_gpu_tmp = completion( TA,  TB,A_gpu,M,K,B_gpu,N,0);

    cudaScale(C_gpu, C_gpu_tmp, M*N, shift_bit);
    cuda_free((float *)C_gpu_tmp);
}

void int_gemm_gpu(int TA, int TB, int M, int N, int K, int ALPHA, 
        char *A, int lda, 
        char *B, int ldb,
        int BETA,
        char *C, int ldc, char shift_bit)
{
    char *A_gpu = int_cuda_make_array(A, (TA ? lda*K:lda*M));
    char *B_gpu = int_cuda_make_array(B, N*K);
    char *C_gpu = int_cuda_make_array(C, ldc*M);

    int_gemm_ongpu(TA, TB, M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc, shift_bit);
    int_cuda_pull_array(C_gpu, C, ldc*M);
    cuda_free((float *)A_gpu);
    cuda_free((float *)B_gpu);
    cuda_free((float *)C_gpu);
}

void int_gemm_bias_ongpu_(int TA, int TB, int M, int N, int K, int ALPHA, 
        char *A_gpu, int lda, 
        char *B_gpu, int ldb,
        int BETA,
        char *C_gpu, int ldc, char shift_bit,char* bias)
{
    int *C_gpu_tmp = completion( TA,  TB,A_gpu,M,K,B_gpu,N,bias);

    cudaScale(C_gpu, C_gpu_tmp, M*N, shift_bit);
    cuda_free((float *)C_gpu_tmp);
}
void int_gemm_bias_gpu(int TA, int TB, int M, int N, int K, int ALPHA, 
        char *A, int lda, 
        char *B, int ldb,
        int BETA,
        char *C, int ldc, char shift_bit,char* bias)
{
    char *A_gpu = int_cuda_make_array(A, (TA ? lda*K:lda*M));
    char *B_gpu = int_cuda_make_array(B, N*K);
    char *C_gpu = int_cuda_make_array(C, ldc*M);

    int_gemm_bias_ongpu_(TA, TB, M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc, shift_bit,bias);

    int_cuda_pull_array(C_gpu, C, ldc*M);
    cuda_free((float *)A_gpu);
    cuda_free((float *)B_gpu);
    cuda_free((float *)C_gpu);
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_gpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
   
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
   
   

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
   
   
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
       
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,192,729,1600); 
       time_gpu(0,0,384,196,1728); 
       time_gpu(0,0,256,196,3456); 
       time_gpu(0,0,256,196,2304); 
       time_gpu(0,0,128,4096,12544); 
       time_gpu(0,0,128,4096,4096); 
     */
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,576,12544); 
    time_gpu(0,0,256,2304,784); 
    time_gpu(1,1,2304,256,784); 
    time_gpu(0,0,512,4608,196); 
    time_gpu(1,1,4608,512,196); 

    return 0;
}
#endif

