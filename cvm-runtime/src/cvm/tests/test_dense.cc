#include <iostream>
#include <string.h>
#include <time.h>
#include "../cuda_ops.h"

void dense(int32_t* dx, int32_t *dw, int32_t* dy, int m , int n, int k, int*db){
  for (uint32_t di = 0; di < m; di++) {
      for (uint32_t oi = 0; oi < n; oi++) {
          int32_t sum = 0;
          for (uint32_t xi = 0; xi < k; xi++) {
              sum += dx[di * k + xi] * dw[oi * k + xi];
          }
		  if(db != nullptr){
			  sum += db[oi];
		  }
          dy[di * n + oi] = sum;
      }
  }
}

void print(int *data, int h, int w, char *label){
    std::cout << label << std::endl;
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            std::cout << data[i*w +j] << " ";
        }
        std::cout << std::endl;
    }
}
int main(){
    int m = 1024;
    int n = 1024;
    int k = 1024;
    int32_t *a = new int[m*k];
    int32_t *b = new int[k*n];
    int32_t *c = new int[m*n];
    int32_t *bias = new int[n];

    for(int i = 0; i < m * k; i++){
        a[i] = i%255;
    }
    for(int i = 0; i < k * n; i++){
        b[i] = 1;
    }
    for(int i = 0; i < n; i++)
        bias[i] = 1;

    dense(a, b, c, m, n, k, bias);
//    print(c, m, n, "cpu");

    int32_t *c2 = new int[m*n];
    cuda_dense(a, b, c2, m, k, n, bias, true);
//    print(c2, m, n, "cuda");
    int ret = memcmp(c, c2, sizeof(int32_t) * m * n);
    std::cout << (ret == 0 ? "pass" : "failed") << std::endl;
}
