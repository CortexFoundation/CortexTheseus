#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "cuda_ops.h"
#include "../common.h"

__global__ void kernel_get_valid_count(const int32_t *input, bool *saved, const int32_t n, const int32_t k, const int32_t score_threshold){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for(uint64_t j = tid; j < n; j+=gridDim.x*blockDim.x){
    const int32_t *row = input + j * k;
    saved[j] = row[1] > score_threshold ? 1 : 0;
  }
}
const char* cuda_get_valid_counts(const int32_t *x_data, int32_t *y_data, int32_t *valid_count_data,
    const int32_t n, const int32_t k,
    const int32_t score_threshold, const int32_t batchs, int& error_code){

  int32_t *host_count = (int32_t*)malloc(sizeof(int32_t) * batchs);
  if(host_count == NULL){
    error_code = ERROR_MALLOC;
    return "malloc error";
  }
  bool *dev_saved = NULL;
  bool* saved = (bool*)malloc(sizeof(bool) * n);
  if(saved == NULL){
    error_code = ERROR_MALLOC;
    goto end;
  }
  cudaError_t status;
  status = cudaMalloc((void**)&dev_saved, sizeof(bool)*n);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }

  for(int32_t i = 0; i < batchs; i++){
    int32_t y_index = 0;
    const int32_t *input = x_data + i * n * k;
    int32_t *output = y_data + i * n * k;

    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_get_valid_count<<<blockSize, threadSize>>>(input, dev_saved, n, k, score_threshold);
    status = cudaMemcpy(saved, dev_saved, sizeof(bool) * n, cudaMemcpyDeviceToHost);
    if(status != cudaSuccess){
      error_code = ERROR_MEMCPY;
      goto end;
    }

    for(int32_t j = 0; j < n; j++){
      const int32_t *row = input + j * k;
      if(saved[j]){
        status = cudaMemcpy(&output[y_index * k], row, k * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        if(status != cudaSuccess){
          error_code = ERROR_MEMCPY;
          goto end;
        }
        y_index += 1;
      }
    }
    host_count[i] = y_index;
    if(y_index < n){
      status = cudaMemset(&output[y_index * k], -1, (n-y_index) * k * sizeof(int32_t));
      if(status != cudaSuccess){
        error_code = ERROR_MEMCPY;
        goto end;
      }
    }
  }

  status = cudaMemcpy(valid_count_data, host_count, sizeof(int32_t) * batchs, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
  }
end:
  if(dev_saved != NULL) cudaFree(dev_saved);
  if(saved != NULL) free(saved);
  if(host_count != NULL) free(host_count);

  /*
     int32_t *h_x = new int32_t[batchs * n * k];
     int32_t *h_vc = new int32_t[batchs];
     int32_t *h_y = new int32_t[batchs * n * k];
     cudaMemcpy(h_x, x_data, batchs*n*k*sizeof(int32_t), cudaMemcpyDeviceToHost);
     get_valid_count(h_x, h_y, h_vc, batchs, n, k, score_threshold);
     cudaMemcpy(y_data, h_y, batchs*n*k*sizeof(int32_t), cudaMemcpyHostToDevice);
     cudaMemcpy(valid_count_data, h_vc, batchs*sizeof(int32_t), cudaMemcpyHostToDevice);
     delete h_x;
     delete h_vc;
     delete h_y;
   */
  return check_cuda_error(cudaGetLastError());
}
__global__ void kernel_get_values_and_keys(
    int32_t* data, const int32_t n, const int32_t k, const int32_t score_index,
    int32_t **values){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < n){
        values[tid] = &data[tid * k];
    }
}

inline __device__ int64_t dev_iou(const int32_t *rect1, const int32_t *rect2, const int32_t format){
    int32_t x1_min = format == FORMAT_CORNER ? rect1[0] : rect1[0] - rect1[2]/2;
    int32_t y1_min = format == FORMAT_CORNER ? rect1[1] : rect1[1] - rect1[3]/2;
    int32_t x1_max = format == FORMAT_CORNER ? rect1[2] : x1_min + rect1[2];
    int32_t y1_max = format == FORMAT_CORNER ? rect1[3] : y1_min + rect1[3];

    int32_t x2_min = format == FORMAT_CORNER ? rect2[0] : rect2[0] - rect2[2]/2;
    int32_t y2_min = format == FORMAT_CORNER ? rect2[1] : rect2[1] - rect2[3]/2;
    int32_t x2_max = format == FORMAT_CORNER ? rect2[2] : x2_min + rect2[2];
    int32_t y2_max = format == FORMAT_CORNER ? rect2[3] : y2_min + rect2[3];

    //x1,x2,y1,y2 precision <= 30
    //sum_arrea precision<=63
    int64_t sum_area = static_cast<int64_t>(x1_max-x1_min) * (y1_max-y1_min) + static_cast<int64_t>(x2_max-x2_min) * (y2_max-y2_min);
    if(sum_area <= 0){
        return 0;
    }

    //w,h precision <= 31
    int32_t w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min));
    int32_t h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min));
    //overlap_area precision <= 62
    int64_t overlap_area = static_cast<int64_t>(h)*w;
    //tmp precision <= 63
    int64_t tmp = (sum_area - overlap_area);
    if(tmp <= 0){
        return 0;
    }
    int64_t max64 = ((uint64_t)1 << 63) - 1;
    if(max64 / 100 < overlap_area){
        tmp /= 100;
    }else{
        overlap_area *= 100;
    }
    int64_t ret = (overlap_area / tmp);//((sum_area - overlap_area)/100));
    return ret;
}

__global__ void kernel_compare_iou(int32_t **rows, int32_t *y_batch,
    const int32_t need_keep, const int32_t k,
    const bool force_suppress, const int32_t id_index, const int32_t coord_start,
    const int32_t iou_threshold,
    bool *removed, 
    int32_t *d_y_index){
    int32_t y_index = 0;
    for(int i = 0; i < need_keep; i++){
      const int32_t *row1 = rows[i];

      if(removed[i] == false && row1[0] >= 0){
        memcpy(&y_batch[y_index*k], row1, k*sizeof(int32_t));
        y_index += 1;
      }
      for(int j = i+1; j < need_keep && !removed[i] && rows[j][0] >= 0; j++){
        const int32_t* row2 = rows[j];
        if(force_suppress || (id_index < 0 || row1[id_index] == row2[id_index])){
          int64_t iou_ret = dev_iou(row1+coord_start, row2+coord_start, FORMAT_CORNER);
          if(iou_ret >= iou_threshold){
            removed[j] = true;
          }
        }
      }
    }
    d_y_index[0] = y_index;
}

const char *cuda_non_max_suppression(int32_t *d_x_data, const int32_t *d_valid_count_data, int32_t *d_y_data, const int32_t batchs, const int32_t n, const int32_t k,
    const int32_t max_output_size, const int32_t iou_threshold, const int32_t topk, 
    const int32_t coord_start, const int32_t score_index, const int32_t id_index, const bool force_suppress, int& error_code){
  int32_t *valid_count_data = (int32_t*)malloc(batchs * sizeof(int32_t));
  int32_t **rows = NULL; 
  bool *removed = NULL;
  int32_t *d_y_index = NULL;
  cudaError_t status;
  if(valid_count_data == NULL){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMemcpy(valid_count_data, d_valid_count_data, batchs*sizeof(int32_t), cudaMemcpyDeviceToHost);
  if(status != cudaSuccess){
    free(valid_count_data);
    error_code = ERROR_MEMCPY;
    goto end;
  }
  status = cudaMalloc((void**)&rows, sizeof(int32_t*) * n);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMalloc((void**)&removed, sizeof(bool) * n);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMalloc((void**)&d_y_index, sizeof(int32_t));
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }

  for(int32_t b = 0; b < batchs; b++){
    int32_t vc = valid_count_data[b];
    if(vc > n){
      goto end;
    }

    int32_t *x_batch = d_x_data + b * n * k;
    int32_t *y_batch = d_y_data + b * n * k;
    if(vc <= 0){
      cudaMemset(y_batch, -1, n * k * sizeof(int32_t));
      goto end;
    }

    if(iou_threshold <= 0){
      status = cudaMemcpy(y_batch, x_batch, vc * k * sizeof(int32_t), cudaMemcpyDeviceToDevice);
      if(status != cudaSuccess){
        error_code = ERROR_MEMCPY;
        goto end;
      }
      status = cudaMemset(y_batch + vc * n * k, -1, (n-vc)*k*sizeof(int32_t));
      if(status != cudaSuccess){
        error_code = ERROR_MEMSET;
        goto end;
      }
    }else{
      const int blockSize = 256;
      const int gridSize = (vc + blockSize - 1) / blockSize;
      kernel_get_values_and_keys<<<gridSize, blockSize>>>(x_batch, vc, k, score_index, rows);
      thrust::stable_sort(thrust::device, rows, rows+vc, [score_index]__device__(const int32_t *a, int32_t *b) -> bool{
            return a[score_index] > b[score_index];
      });

      if(topk > 0 && topk < vc){
        for(int i = 0; i < vc - topk; i++){
          status = cudaMemset(rows[i+topk], -1, k*sizeof(int32_t));
          if(status != cudaSuccess){
            error_code = ERROR_MEMSET;
            goto end;
          }
        }
      }

      status = cudaMemset(removed, false, sizeof(bool)*vc);
      if(status != cudaSuccess){
        error_code = ERROR_MEMSET;
        goto end;
      }
      int need_keep = ((topk >= 0 && topk < vc) ? topk : vc);
      if(need_keep < vc){
        status = cudaMemset(removed + need_keep, true, (vc-need_keep)*sizeof(bool));
        if(status != cudaSuccess){
          error_code = ERROR_MEMSET;
          goto end;
        }
      }
      int32_t y_index;
      kernel_compare_iou<<<1,1>>>(rows, y_batch, need_keep, k, force_suppress,
          id_index, coord_start, iou_threshold, removed, d_y_index);
      status = cudaMemcpy(&y_index, d_y_index, sizeof(int32_t), cudaMemcpyDeviceToHost);
      if(status != cudaSuccess){
        error_code = ERROR_MEMCPY;
        goto end;
      }

      if(y_index < n){
        status = cudaMemset(&y_batch[y_index*k], -1, (n-y_index)*k*sizeof(int32_t));
        if(status != cudaSuccess){
          error_code = ERROR_MEMSET;
          goto end;
        }
      }
    } 
    if(max_output_size > 0){
      int j = 0;
      for(int i = 0; i < vc; i++){
        if(j == max_output_size){
          status = cudaMemset(y_batch + i * k, -1, k * sizeof(int32_t));
          if(status != cudaSuccess){
            error_code = ERROR_MEMSET;
            goto end;
          }
        }else{
          j += 1;
        }
      }
    }
  }
end:
  if(valid_count_data != NULL) free(valid_count_data);
  if(rows != NULL) cudaFree(rows);
  if(d_y_index != NULL) cudaFree(d_y_index);
  if(removed != NULL) cudaFree(removed);
  return check_cuda_error(cudaGetLastError());
}

//const char *cuda_non_max_suppression(int32_t *d_x_data, const int32_t *d_valid_count_data, int32_t *d_y_data, const int32_t batchs, const int32_t n, const int32_t k,
//    const int32_t max_output_size, const int32_t iou_threshold, const int32_t topk, 
//    const int32_t coord_start, const int32_t score_index, const int32_t id_index, const bool force_suppress, int& error_code){
//  int32_t *x_data = NULL, *valid_count_data = NULL, *y_data = NULL;
//  x_data = (int32_t*)malloc(sizeof(int32_t) * batchs*n*k);//new int32_t[batchs * n * k];
//  valid_count_data = (int32_t*)malloc(sizeof(int32_t)*batchs);//new int32_t[batchs];
//  y_data = (int32_t*)malloc(sizeof(int32_t) *batchs*n*k);//new int32_t[batchs * n * k];
//  int ret = 0;
//  if(x_data == NULL || valid_count_data == NULL || y_data == NULL){
//    error_code = ERROR_MALLOC;
//    goto end;
//  }
//  cudaError_t status;
//  status = cudaMemcpy(x_data, d_x_data, batchs*n*k*sizeof(int32_t), cudaMemcpyDeviceToHost);
//  if(status != cudaSuccess){
//    error_code = ERROR_MEMCPY;
//    goto end;
//  }
//  status = cudaMemcpy(valid_count_data, d_valid_count_data, batchs*sizeof(int32_t), cudaMemcpyDeviceToHost);
//  if(status != cudaSuccess){
//    error_code = ERROR_MEMCPY;
//    goto end;
//  }
//
//  ret = non_max_suppression(
//      x_data, valid_count_data, y_data, batchs, n, k,
//      max_output_size, iou_threshold, topk, coord_start, score_index, id_index, force_suppress);
//
//  status = cudaMemcpy(d_y_data, y_data, batchs * n * k * sizeof(int32_t), cudaMemcpyHostToDevice);
//  if(status != cudaSuccess){
//    error_code = ERROR_MEMCPY;
//  }
//
//end:
//  if(x_data != NULL)
//    free(x_data);
//  if(valid_count_data != NULL)
//    free(valid_count_data);
//  if(y_data != NULL)
//    free(y_data);
//  if(ret < 0){
//    return "the valid count must less than the number of box";
//  }
//  return check_cuda_error(cudaGetLastError());
//}

