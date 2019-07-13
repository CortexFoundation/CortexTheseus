#ifndef NMS_H
#define NMS_H

#define FORMAT_CORNER 1
#define FORMAT_CENTER 2
int64_t iou(const int32_t *rect1, const int32_t *rect2, const int32_t format);

void get_valid_count(const int32_t *x_data, int32_t *y_data, int32_t *valid_count_data, const int32_t batchs,
        const int32_t n, const int32_t k, const int32_t score_threshold);

int non_max_suppression(int32_t *x_data, const int32_t *valid_count_data, int32_t *y_data, const int32_t batchs, const int32_t n, const int32_t k,
        const int32_t max_output_size, const int32_t iou_threshold, const int32_t topk,
        const int32_t coord_start, const int32_t score_index, const int32_t id_index, const bool force_suppress);

#endif
