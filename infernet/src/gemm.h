#ifndef GEMM_H
#define GEMM_H

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
        
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

#ifdef GPU
void cudaScale(char* output, int* input, int size, char shift_bit);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

void int_gemm_gpu(int TA, int TB, int M, int N, int K, int ALPHA, 
        char *A, int lda, 
        char *B, int ldb,
        int BETA,
        char *C, int ldc, char shift_bit);

void int_gemm_bias_gpu(int TA, int TB, int M, int N, int K, int ALPHA, 
        char *A, int lda, 
        char *B, int ldb,
        int BETA,
        char *C, int ldc, char shift_bit,char* bias);

void rmPadding_gpu(int* h_C,int* h_C_new,int new_N,int new_M,int old_N,int old_M);
void add_bias_gpu_fc(int* h_C,int size, char* bias);
void addPaddingA_gpu(char* h_A_new,char* h_A,int new_N,int new_M,int old_N,int old_M, int T);
void addPaddingB_gpu(char* h_B_new,char* h_B,int new_N,int new_M,int old_N,int old_M,int T);

#endif
#endif
