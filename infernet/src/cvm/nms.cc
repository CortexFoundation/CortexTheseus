#include <iostream>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include <string.h>
#include "nms.h"

int64_t iou(const int32_t *rect1, const int32_t *rect2, const int32_t format){
    int32_t x1_min = format == FORMAT_CORNER ? rect1[0] : rect1[0] - rect1[2]/2;
    int32_t y1_min = format == FORMAT_CORNER ? rect1[1] : rect1[1] - rect1[3]/2;
    int32_t x1_max = format == FORMAT_CORNER ? rect1[2] : x1_min + rect1[2];
    int32_t y1_max = format == FORMAT_CORNER ? rect1[3] : y1_min + rect1[3];

    int32_t x2_min = format == FORMAT_CORNER ? rect2[0] : rect2[0] - rect2[2]/2;
    int32_t y2_min = format == FORMAT_CORNER ? rect2[1] : rect2[1] - rect2[3]/2;
    int32_t x2_max = format == FORMAT_CORNER ? rect2[2] : x2_min + rect2[2];
    int32_t y2_max = format == FORMAT_CORNER ? rect2[3] : y2_min + rect2[3];

    int64_t sum_area = static_cast<int64_t>(std::abs(x1_max-x1_min)) * std::abs(y1_max-y1_min) + static_cast<int64_t>(std::abs(x2_max-x2_min)) * std::abs(y2_max-y2_min);

    if(x1_min > x2_max || x1_max < x2_min || y1_min > y2_max || y1_max < y2_min) return 0;
    int32_t w = std::min(x1_max, x2_max) - std::max(x1_min, x2_min);
    int32_t h = std::min(y1_max, y2_max) - std::max(y1_min, y2_min);
    int64_t overlap_area = static_cast<int64_t>(h)*w;
    int64_t tmp = (sum_area - overlap_area) / 100;
    if(tmp <= 0){
        return 100;
    }
    int64_t ret = (overlap_area / ((sum_area - overlap_area)/100));
    return ret;
}

void get_valid_count(const int32_t *x_data, int32_t *y_data, int32_t *valid_count_data, const int32_t batchs,
        const int32_t n, const int32_t k, const int32_t score_threshold){
    for(int32_t i = 0; i < batchs; i++){
        int32_t y_index = 0;
        const int32_t *input = x_data + i * n * k;
        int32_t *output = y_data + i * n * k;
        for(int32_t j = 0; j < n; j++){
            const int32_t *row = input + j * k;
            if(row[1] > score_threshold){
                memcpy(&output[y_index * k], row, k * sizeof(int32_t));
                y_index += 1;
            }
        }
        valid_count_data[i] = y_index;
        if(y_index < n){
            memset(&output[y_index * k], -1, (n-y_index) * k * sizeof(int32_t));
        }
    }
}

void non_max_suppression(int32_t *x_data, const int32_t *valid_count_data, int32_t *y_data, const int32_t batchs, const int32_t n, const int32_t k,
        const int32_t max_output_size, const int32_t iou_threshold, const int32_t topk,
        const int32_t coord_start, const int32_t score_index, const int32_t id_index, const bool force_suppress){
    for(int32_t b = 0; b < batchs; b++){
        int32_t vc = valid_count_data[b];
        std::vector<int32_t*> rows(n);
        int32_t *x_batch = x_data + b * n * k;
        int32_t *y_batch = y_data + b * n * k;

        for (int i = 0; i < n; i++) {
            rows[i] = x_batch + i * k;
        }
        for(int i = vc; i < n; i++){
            memset(rows[i], -1, k * sizeof(int32_t));
        }
        auto score_idx_local = score_index;
        std::sort(rows.begin(), rows.end(), [&score_idx_local](const int32_t* a, const int32_t* b){
                return a[score_idx_local] > b[score_idx_local];
        });
        if(topk > 0 && topk < vc){
            for(int i = 0; i < vc - topk; i++){
                memset(rows[i+topk], -1, k * sizeof(int32_t));
            }
        }

        std::vector<bool> removed(n, false);
        int start_i = ((topk >= 0 && topk < vc) ? topk : vc);
        for(int i = start_i; i < n; i++){
            removed[i] = true;
        }

        int32_t y_index = 0;
        for(int i = 0; i < vc; i++){
            int32_t *row1 = rows[i];

            if(removed[i] == false){
                memcpy(&y_batch[y_index*k], row1, k*sizeof(int32_t));
                y_index += 1;
            }
            for(int j = i+1; j < n && !removed[i] && iou_threshold > 0; j++){
                int32_t* row2 = rows[j];
                if(force_suppress || (id_index < 0 || row1[id_index] == row2[id_index])){
                    if(iou(row1+coord_start, row2+coord_start, FORMAT_CORNER) > iou_threshold){
                        removed[j] = true;
                    }
                }
            }
        }
        if(y_index < n){
            memset(&y_batch[y_index*k], -1, (n - y_index) * k * sizeof(int32_t));
        }
        if(max_output_size > 0){
            if(max_output_size < y_index){
                memset(&y_batch[max_output_size * k], -1, (y_index - max_output_size) * k * sizeof(int32_t));
            }
        }
    }
}
