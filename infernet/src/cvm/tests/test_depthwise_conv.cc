#include <iostream>
#include <string.h>
#include <time.h>
#include "../cuda_ops.h"
inline void depthwise_conv2d_cpu(
        int32_t *x_data, int32_t n_batch, int32_t in_channels, int32_t x_h, int32_t x_w,
        int32_t *w_data, int32_t filter_h, int32_t filter_w,
        int32_t *y_data, int32_t out_channels, int32_t o_h, int32_t o_w,
        int32_t *b_data,
        int32_t padding, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
        int32_t groups){
//    assert(in_channels == out_channels);
//    assert(in_channels == groups);

    for(int n = 0; n < n_batch; ++n){
        for(int c = 0; c < in_channels; ++c){
            for(int h = 0; h < o_h; ++h){
                for(int w = 0; w < o_w; ++w){
                    int32_t sum = 0;
                    for(int fh = 0; fh < filter_h; ++fh){
                        for(int fw = 0; fw < filter_w; ++fw){
                            int th = h * stride_h + fh*dilation_h - padding;
                            int tw = w * stride_h + fw*dilation_w - padding;
                            if(th < 0 || tw < 0 || th >= x_h || tw >= x_w)
                                continue;
                            sum += x_data[n * in_channels * x_h * x_w + c * x_h * x_w + th * x_w + tw]
                                * w_data[c * filter_h * filter_w + fh * filter_w + fw];
                        }
                    }
                    y_data[n * in_channels * o_h * o_w + c * o_h * o_w + h * o_w + w] = sum + b_data[c];
                }
            }
        }
    }
}
void print(int* data, int n, int c, int h, int w){
    for(int in = 0; in < n; in++){
    for(int i = 0; i < c; i++){
        for(int j = 0; j < h; j++){
            for(int k = 0; k < w; k++){
                std::cout << data[in*c*h*w + i*h*w+j*w+k] << " ";
            }
            std::cout << std::endl;
        }
    }
    }
}

int main(){
    int i_n = 2;
    int i_c = 64;
    int i_h = 32;
    int i_w = i_h;
    int f_h = 3;
    int f_w = f_h;
    int o_c = i_c;
    int padding = 1;
    int stride = 2;
    int dilation_h = 2;
    int dilation_w = 2;
    int tmp_f_h = (f_h - 1) * dilation_h + 1; // for dilation, to be optimized
    int tmp_f_w = (f_w - 1) * dilation_w + 1;
    int o_h = (i_h + 2 * padding - tmp_f_h) / stride + 1;
    int o_w = (i_w + 2 * padding - tmp_f_w) / stride + 1;
    size_t s_i = i_n * i_c * i_h * i_w;
    size_t s_f = i_n * i_c * f_h * f_w;
    size_t s_o = i_n * o_c * o_h * o_w;
    int *input = new int[s_i];
    int *filter = new int[s_f];
    int *b_data = new int[o_c];
    int *output = new int[s_o];
    for(int i = 0; i < s_i; i++)
        input[i] = i%255 - 127;
    for(int i = 0; i < s_f; i++)
        filter[i] = 1;
    for(int i = 0; i < o_c; i++)
        b_data[i] = 1;
//    print(input, i_c, i_h, i_w);
    clock_t start = clock();
    depthwise_conv2d_cpu(input, i_n, i_c, i_h, i_w,
        filter, f_h, f_w,
        output, o_c, o_h, o_w,
        b_data,
        padding, stride, stride, dilation_h, dilation_w,
        i_n);
    clock_t end = clock();
    std::cout << "cpu time: " << end-start << std::endl;
//    print(output, i_n, o_c, o_h, o_w);

    int* output2 = new int[s_o];
    cuda_depthwise_conv2d(
        input, i_n, i_c, i_h, i_w,
        filter, i_n, i_c, f_h, f_w,
        b_data,
        padding, padding,
        stride, stride,
        dilation_h, dilation_w,
        i_c,
        output2, i_n, o_c, o_h, o_w, 0, true);

    clock_t gpu_end = clock();
    std::cout << "gpu all time: " << gpu_end - end << std::endl;
//    print(output2, i_n, o_c, o_h, o_w);
    int ret = memcmp(output, output2, sizeof(int) * s_o);
    std::cout << (ret == 0 ? "pass" : "failed") << std::endl;
    return 0;
}
