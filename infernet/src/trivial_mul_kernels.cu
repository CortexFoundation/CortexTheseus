#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "im2col.h"
#include "cuda.h"
#include "trivial_mul.h"
}

__global__ void matrixMultiplyShared(char *A, char *B, int *C, int numARows,
    int numAColumns, int numBRows,
    int numBColumns, int numCRows,
    int numCColumns) {
//@@ Insert code to implement matrix multiplication here
//@@ You have to use shared memory for this MP

__shared__ char sharedM[TILE_WIDTH][TILE_WIDTH];
__shared__ char sharedN[TILE_WIDTH][TILE_WIDTH];
int bx = blockIdx.x;
int by = blockIdx.y;
int tx = threadIdx.x;
int ty = threadIdx.y;
int row = by*TILE_WIDTH + ty;
int col = bx*TILE_WIDTH + tx;
int v = 0;

for (int i = 0; i < (int)(ceil((float)numAColumns/TILE_WIDTH)); i++)
{
if (i*TILE_WIDTH + tx < numAColumns && row < numARows)
sharedM[ty][tx] = A[row*numAColumns + i*TILE_WIDTH + tx];
else
sharedM[ty][tx] = 0;

if (i*TILE_WIDTH + ty < numBRows && col < numBColumns)
sharedN[ty][tx] = B[(i*TILE_WIDTH + ty)*numBColumns + col];
else
sharedN[ty][tx] = 0;
__syncthreads();

for(int j = 0; j < TILE_WIDTH; j++)
v += sharedM[ty][j] * sharedN[j][tx];
__syncthreads();
}

if (row < numCRows && col < numCColumns)
C[row*numCColumns + col] = v;

}
extern "C" void gemmExt(char *A, char *B, int *C, int numARows,
    int numAColumns, int numBRows,
    int numBColumns, int numCRows,
    int numCColumns)
{
// unsigned int A_size = numARows * numAColumns * sizeof(float);
// unsigned int B_size = numBRows * numBColumns * sizeof(float);
// unsigned int C_size = numCRows * numCColumns * sizeof(float);
dim3 DimGrid(ceil(numCColumns / 32.0), ceil(numCRows / 32.0), 1);
dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

matrixMultiplyShared<<< DimGrid, DimBlock >>>(A, B, C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
cudaDeviceSynchronize();
return ;
}