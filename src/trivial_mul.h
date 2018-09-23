
#define TILE_WIDTH 32  //block size ,each thread to calucate each block
// void gemmExtt(char *A, char *B, int *C, int numARows,
//                                      int numAColumns, int numBRows,
//                                      int numBColumns, int numCRows,
//                                      int numCColumns);
void gemmExt(char *A, char *B, int *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns);