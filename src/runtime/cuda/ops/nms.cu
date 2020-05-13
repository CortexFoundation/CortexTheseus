#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cvm/top/nn.h>
#include "cuda_ops.h"

namespace cvm{
namespace runtime{

template<int score_index, int BS>
__global__ void kernel_get_count(const int32_t batchs, const int32_t n, const int32_t K, const int32_t *inputs, int32_t *valid_count, const int32_t score_threshold, int32_t *all_count){
  const int lid = threadIdx.x;
  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y;
  const int gidx = lid + BS * bidx;  
  const int batch = bidy; 
  __shared__ int32_t share_box[BS][32];
  __shared__ int32_t count;
  share_box[lid][score_index] = -1;
  if(lid == 0) {
    count = 0;
  }
  __syncthreads();
  int x_i = batch * n * K + gidx * K; 
  if(gidx < n){
    for(int i = 0; i < K; i++)
      share_box[lid][i] = inputs[x_i + i];
  }

  if(share_box[lid][score_index] > score_threshold){
    atomicAdd(&count, 1);
  }
  __syncthreads();

  for(int i = lid+1; i < gridDim.x; i+=blockDim.x){
    atomicAdd(&all_count[batch * n + bidx + i], count);
  }
  if(lid == blockDim.x -1){
    atomicAdd(valid_count + batch, count);
  }
}

template<int score_index, int BS>
__global__ void kernel_get_data(const int32_t batchs, const int32_t n, const int32_t K, const int32_t *inputs, int32_t *y, int32_t *valid_count, const int32_t score_threshold, int32_t *all_count){
  const int lid = threadIdx.x;
  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y;
  const int gidx = lid + BS * bidx;  
  const int batch = bidy; 
  __shared__ int32_t share_box[BS][32];
  __shared__ int32_t count;
  __shared__ int32_t start_index;
  __shared__ int32_t tmp_indexs[BS];
  for(int i = 0; i < K; i++){
    share_box[lid][i] = -1;
  }
  if(lid == 0) {
    count = 0;
    start_index = all_count[batch * n + bidx];
  }
  tmp_indexs[lid] = 0;
  int x_i = batch * n * K + gidx * K; 
  __syncthreads();

  if(gidx < n){
    for(int i = 0; i < K; i++)
      share_box[lid][i] = inputs[x_i + i];
  }
  if(share_box[lid][score_index] <= score_threshold){
      share_box[lid][0] = -1;
  }
  else{
    atomicAdd(&count, 1);
    for(int i = lid+1; i < BS; i++){
      atomicAdd(tmp_indexs + i, 1);
    }
  }

  __syncthreads();

  if(share_box[lid][0] != -1){
    for(int k = 0; k < K; ++k)
      y[batch * n * K + (start_index + tmp_indexs[lid]) * K + k] = share_box[lid][k];
  }
}

__global__ void kernel_get_values_and_keys(
    int32_t* data, const int32_t n, const int32_t k, const int32_t score_index,
    int32_t **values){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < n){
        values[tid] = &data[tid * k];
    }
}
__global__ void kernel_get_count_sorted(int32_t **rows, const int n, const int K, int32_t *valid_count, const int score_threshold, int32_t* output){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ int count;
  if(threadIdx.x == 0)  count = 0;
  __syncthreads();
  if(tid < n){
    if(rows[tid][1] > score_threshold)  {
      atomicAdd(&count, 1);
      for(int k = 0; k < K; k++)
        output[tid * K + k] = rows[tid][k];
    }else{
      for(int k = 0; k < K; k++)
        output[tid * K + k] = -1;//rows[tid][k];
    }
  }
  __syncthreads();
  atomicAdd(valid_count, count);
}

const char* cuda_get_valid_counts(int32_t *x_data, int32_t *y_data, int32_t *valid_count_data,
   const int32_t n, const int32_t k,
   const int32_t score_threshold, const int32_t batchs, int32_t *ext_space, int& error_code){

  int32_t *all_count = ext_space;
  cudaMemset(y_data, -1, sizeof(int32_t) * batchs * n * k);
  cudaMemset(valid_count_data, 0, sizeof(int32_t) * batchs);
  cudaMemset(all_count, 0, sizeof(int32_t) * batchs * n);

  const int bsize = 128;
  dim3 gsize = dim3((n+bsize-1)/bsize, batchs, 1);
  kernel_get_count<1, bsize><<<gsize, bsize>>>(batchs, n, k, x_data, valid_count_data, score_threshold, all_count);
  kernel_get_data<1, bsize><<<gsize, bsize>>>(batchs, n, k, x_data, y_data, valid_count_data, score_threshold, all_count);

 // int32_t **rows = (int32_t**)ext_space; 
 // int blockSize = 256;
 // int gridSize = (n + blockSize - 1) / blockSize;
 // int score_index = 1;
 // for(int i = 0; i < batchs; i++){
 //   int32_t *x_batch = x_data + i * n * k;
 //   kernel_get_values_and_keys<<<gridSize, blockSize>>>(x_batch, n, k, score_index, rows);
 //   thrust::stable_sort(thrust::device, rows, rows+n, [score_index]__device__(const int32_t *a, int32_t *b) -> bool{
 //       return a[1] > b[1];
 //       });
 //   kernel_get_count_sorted<<<gridSize, blockSize>>>(rows, n, k, valid_count_data + i, score_threshold, y_data+ i * n*k);
 // }

  print_to_file(x_data, batchs * n * k, "get_valid_count_x.txt");
  print_to_file(y_data, batchs * n * k, "get_valid_count_y.txt");

  return ""; 
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

//#define BS 64 // the block size(BS, BS)
template<const int32_t BS, const int32_t id_index, const int32_t coord_start, int K>
__global__ void kernel_cal_all_iou(int32_t **rows, bool *removed, const int n, int32_t iou_threshold){
  int bidx = blockIdx.x;
  int bidy = blockIdx.y;
  int lidx = threadIdx.x;
  int lidy = threadIdx.y;
  int gidx = lidx + bidx * BS;
  int gidy = lidy + bidy * BS;
  __shared__ int32_t share_box1[BS][K];
  __shared__ int32_t share_box2[BS][K];
  int index1 = bidy * BS + lidx;
  if(lidy == 0 && index1 < n){
#pragma unroll
    for(int i = 0; i < K; i++)
      share_box1[lidx][i] = rows[index1][i];
  }
  if(lidy == 1 && gidx < n){
#pragma unroll
    for(int i = 0; i < K; i++)
      share_box2[lidx][i] = rows[gidx][i];
  }
  __syncthreads();

  if(gidx < n && gidy < n && gidy < gidx){
    int64_t iou_ret = dev_iou(&share_box1[lidy][coord_start], &share_box2[lidx][coord_start], FORMAT_CORNER); 
    if(iou_ret >= iou_threshold){
      removed[gidy * n + gidx] = true; 
    }
  }
}

template<bool force_suppress, const int32_t id_index, const int32_t coord_start, int K>
__global__ void kernel_compare_iou_opt(const int32_t idx_max, const int32_t n_max, bool* removed, int32_t *y_batch, int32_t **rows, int32_t *num_y){
  int yn = 0;
  const int32_t removed_n = max(n_max, idx_max);

  if(n_max < 8*1024){
    __shared__ int yindex[1024*8];
    for(int i = 0; yn < n_max && i < idx_max; i++){
      int32_t row[K];
#pragma unroll
      for(int k = 0; k < K; k++){
        row[k] = rows[i][k];
      }
      if(row[id_index] < 0) continue;

      int j = 0;
      for(; j < yn; j++){
        bool flag = removed[yindex[j] * removed_n + i];
        if(force_suppress || y_batch[j*K + id_index] == row[id_index]){
            if(flag) {
              break;
            }
        } }
      if(j == yn){
#pragma unroll
        for(int k = 0; k < K; k++){
          y_batch[yn * K + k] = row[k];
        }
        yindex[yn] = i;
        ++yn;
      } 
    }
  }else{
    for(int i = 0; yn < n_max && i < idx_max; i++){
      int32_t row[K];
#pragma unroll
      for(int k = 0; k < K; k++){
        row[k] = rows[i][k];
      }
      if(row[id_index] < 0) continue;

      int j = 0;
      for(; j < i; j++){
        bool preflag = removed[j];
        if(!preflag && (force_suppress || rows[j][id_index] == row[id_index])){
          bool flag = removed[j * removed_n + i];
            if(flag) {
              removed[i] = true;
              break;
            }
        }
      }
      if(j == i){
#pragma unroll
        for(int k = 0; k < K; k++){
          y_batch[yn * K + k] = row[k];
        }
        ++yn;
      } 
    }
  }
  *num_y = yn;
}

template<int BS, bool force_suppress, const int32_t id_index, const int32_t coord_start, int K>
__global__ void kernel_compare_iou_opt2(const int32_t idx_max, const int32_t n_max,int32_t *y_batch, int32_t **rows, int32_t *num_y, int32_t iou_threshold){
  int lid = threadIdx.x;
  __shared__ int32_t share_box1[BS][K];
  __shared__ int32_t share_box2[BS][K];
  __shared__ int32_t yn;

  if(lid == 0) yn = 0;
  __syncthreads();
  for(int i = 0; i < idx_max && yn < n_max; i+= BS){
    for(int k = 0; k < K; k++){
      share_box1[lid][k] = -1;
    }
    if(i + lid < idx_max){
      for(int k = 0; k < K; k++){
        share_box1[lid][k] = rows[i+lid][k]; 
      }
    }
    __syncthreads();
    for(int j = 0; j < yn && yn < n_max; j+=BS){
      for(int k = 0; k < K; k++){
        share_box2[lid][k] = -1;
      }
      if(lid + j < yn){
        for(int k = 0; k < K; k++){
          share_box2[lid][k] = y_batch[(lid+j) * K + k];
        }
      }
      __syncthreads();
      for(int l = 0; l < BS && share_box1[lid][0] !=-1; l++){
        if(share_box2[l][0] == -1) continue;
        if(force_suppress || share_box1[lid][id_index] == share_box2[l][id_index]){
          int64_t iou_ret = dev_iou(&share_box1[lid][coord_start], &share_box2[l][coord_start], FORMAT_CORNER); 
          if(iou_ret >= iou_threshold) {
            for(int k = 0; k < K; k++){
              share_box1[lid][k] = -1;
            }
            break;
          }
        }
      }
      __syncthreads();
    }
    __syncthreads();
    for(int j = 0; j < BS; j++){
      if(lid < j && !share_box1[lid][0] != -1 && share_box1[j][0] != -1){
        if(force_suppress || share_box1[lid][id_index] == share_box1[j][id_index]){
          int64_t iou_ret = dev_iou(&share_box1[lid][coord_start], &share_box1[j][coord_start], FORMAT_CORNER); 
          if(iou_ret >= iou_threshold) {
            share_box1[j][0] = -1;
          }
        }
      }
      __syncthreads();
    }
    if(lid == 0){
      for(int l = 0; l < BS; l++){
        if(share_box1[l][0] != -1){
          for(int k = 0; k < K; k++)
            y_batch[yn * K + k] = share_box1[l][k];
          ++yn;
        }
      }    
    }
    __syncthreads();
  }
  if(lid == 0) *num_y = yn;
}

const char *cuda_non_max_suppression(int32_t *d_x_data, const int32_t *d_valid_count_data, int32_t *d_y_data, const int32_t batchs, const int32_t n, const int32_t k,
    const int32_t max_output_size, const int32_t iou_threshold, const int32_t topk, 
    const int32_t coord_start, const int32_t score_index, const int32_t id_index, const bool force_suppress, int32_t *ext_space, int& error_code){
  int32_t *valid_count_data = (int32_t*)malloc(batchs * sizeof(int32_t));
  int32_t **rows = (int32_t**)ext_space; 
  int32_t *d_y_index = ext_space + n * (sizeof(int32_t*) / sizeof(int32_t));
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

  for(int32_t b = 0; b < batchs; b++){
    int32_t vc = valid_count_data[b];

    vc = std::max(std::min(vc, n), 0);

    int32_t *x_batch = d_x_data + b * n * k;
    int32_t *y_batch = d_y_data + b * n * k;
    if(vc <= 0){
      cudaMemset(y_batch, -1, n * k * sizeof(int32_t));
      goto end;
    }

    int32_t n_max = vc;
    if(max_output_size >= 0) n_max = std::min(n_max, max_output_size);
    int32_t idx_max = vc;
    if(topk >= 0) idx_max = std::min(idx_max, topk);

    int blockSize = 256;
    int gridSize = (vc + blockSize - 1) / blockSize;
    kernel_get_values_and_keys<<<gridSize, blockSize>>>(x_batch, vc, k, score_index, rows);
    thrust::stable_sort(thrust::device, rows, rows+vc, [score_index]__device__(const int32_t *a, int32_t *b) -> bool{
        return a[score_index] > b[score_index];
    });

    if(force_suppress){
      kernel_compare_iou_opt2<128, true, 0, 2, 6><<<1, 128>>>(idx_max, n_max, y_batch, rows, d_y_index, iou_threshold);
    }
    else{ 
      kernel_compare_iou_opt2<128, false, 0, 2, 6><<<1,128>>>(idx_max, n_max, y_batch, rows, d_y_index, iou_threshold);
    }
    int32_t yn = 0;
    cudaMemcpy(&yn, d_y_index, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemset(y_batch + yn * k, -1, (n - yn) * k * sizeof(int32_t));
  } 
end:
  if(valid_count_data != NULL) free(valid_count_data);
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

}
}
