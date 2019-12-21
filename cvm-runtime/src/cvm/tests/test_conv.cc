#include <iostream>
#include <string.h>
#include <time.h>
#include "../cuda_ops.h"

void conv_cpu(int* x_data, int n_batch, int x_h, int x_w, int in_channels, int *w_data, int filter_h, int filter_w,
        int *b_data,
        int *y_data, int o_h, int o_w, int out_channels,
        int stride_h, int stride_w,
        int padding_h, int32_t padding_w,
        int dilation_h, int dilation_w
        ){
#define GETX(n, c, h, w) x_data[(n) * in_channels * x_h * x_w + (c) * x_h * x_w + (h) * x_w + (w)]
#define GETW(o, i, h, w) w_data[(o) * in_channels * filter_h * filter_w + (i) * filter_h * filter_w + (h) * filter_w + (w)]
#define GETY(n, c, h, w) y_data[(n) * out_channels * o_h * o_w + (c) * o_h * o_w + (h) * o_w + (w)]
    auto calc_func = [&](int n, int k, int p, int q) {
        int y_sum = 0;
        for (int c = 0; c < in_channels; ++c) {
            for (int r = 0; r < filter_h; ++r) {
                for (int s = 0; s < filter_w; ++s) {
                    auto tp = p * stride_h + r*dilation_h - padding_h;
                    auto tq = q * stride_w + s*dilation_w - padding_w;
                    if (tp < 0 || tq < 0 || tp >= x_h || tq >= x_w)
                        continue;
                    y_sum += GETX(n, c, tp, tq) * GETW(k, c, r, s);
                }
            }
        }
        return y_sum;

    };
    for (int n = 0; n < n_batch; ++n) {
        for (int k = 0; k < out_channels; ++k) {
            for (int p = 0; p < o_h; ++p) {
                for (int q = 0; q < o_w; ++q) {
                    GETY(n, k, p, q) = b_data[k] + calc_func(n, k, p, q);
                }
            }
        }
    }
}

#define BS 16
//TODO padding != 0
void conv_cpu_v2(int* x_data, int n_batch, int x_h, int x_w, int in_channels,
        int *w_data, int filter_h, int filter_w,
        int *b_data,
        int *y_data, int o_h, int o_w, int out_channels,
        int stride_h, int stride_w,
        int padding
        ){
    int tmp_o_h = x_h + 2 * padding - filter_h + 1;
    int tmp_o_w = x_w + 2 * padding - filter_w + 1;
#define GETX(n, c, h, w) x_data[(n) * in_channels * x_h * x_w + (c) * x_h * x_w + (h) * x_w + (w)]
#define GETW(o, i, h, w) w_data[(o) * in_channels * filter_h * filter_w + (i) * filter_h * filter_w + (h) * filter_w + (w)]
    for(int n = 0; n < n_batch; ++n){
        for(int oc = 0; oc < out_channels; ++oc){
            int tmpb = b_data[oc];
            for(int oh = 0; oh < tmp_o_h; oh += BS){
                for(int ow = 0; ow < tmp_o_w; ow += BS){
                    int32_t sum[BS][BS] = {0};
                    int min_y = oh+BS <= tmp_o_h ? BS : tmp_o_h%BS;
                    int min_x = ow+BS <= tmp_o_w ? BS : tmp_o_w%BS;
                    for(int ic = 0; ic < in_channels; ++ic){
                        int32_t bufX[BS+12][BS+12]; //filter_h <= 11
                        int32_t bufF[12][12]; //filter_h <= 11
                        for(int i = 0; i < BS; i++){
                            for(int j = 0; j < BS; j++){
                                if(oh+i-padding < 0 || ow+j-padding<0 || oh+i-padding>= x_h || ow+j-padding>=x_w)
                                    bufX[i][j] = 0;
                                else
                                    bufX[i][j] = x_data[n*in_channels*x_h*x_w + ic*x_h*x_w + (oh+i-padding)*x_w + ow+j-padding];
                                if(i < filter_h-1){
                                    for(int ti = i; ti < filter_h-1; ti+=min_y){
                                        if(ti+oh+min_y-padding<0 || ow+j-padding < 0 || ti+oh+min_y-padding>=x_h || ow+j-padding>=x_w){
                                            bufX[ti+min_y][j] = 0;
                                        }
                                        else{
                                            bufX[ti+min_y][j] = x_data[n*in_channels*x_h*x_w + ic*x_h*x_w + (ti+oh+min_y-padding)*x_w + ow+j-padding];
                                        }
                                    }
                                }
                                if(j < filter_w-1){
                                    for(int tj = j; tj < filter_w-1; tj+=min_x){
                                        if(oh+i-padding<0 || tj+ow+min_x-padding<0 || oh+i-padding>=x_h || tj+ow+min_x-padding>=x_w)
                                            bufX[i][tj+min_x] = 0;
                                        else
                                            bufX[i][tj+min_x] = x_data[n*in_channels*x_h*x_w + ic*x_h*x_w + (oh+i-padding)*x_w + tj+ow+min_x-padding];
                                    }
                                }
                                if(i < filter_h-1 && j < filter_w-1){
                                    for(int ti = i; ti < filter_h-1; ti+=min_y){
                                        for(int tj = j; tj < filter_w-1; tj+=min_x){
                                            if(ti+oh+min_y-padding<0 || tj+ow+min_x-padding<0 || ti+oh+min_y-padding>=x_h || tj+ow+min_x-padding>=x_w)
                                                bufX[ti+min_y][tj+min_x] = 0;
                                            else
                                                bufX[ti+min_y][tj+min_x] = x_data[n*in_channels*x_h*x_w + ic*x_h*x_w + (ti+oh+min_y-padding)*x_w + tj+ow+min_x-padding];
                                        }
                                    }
                                }
                                if(i < filter_h && j < filter_w){
                                    for(int ti = i; ti < filter_h; ti+= min_y){
                                        for(int tj = j; tj < filter_w; tj+=min_x){
                                            bufF[ti][tj] = w_data[oc*in_channels*filter_h*filter_w + ic*filter_h*filter_w + ti*filter_w + tj];
                                        }
                                    }
                                }
                            }
                        }
                        for(int i = 0; i < min_y; i++){
                            for(int j = 0; j < min_x; j++){
                                for(int fh = 0; fh < filter_h; ++fh){
                                    for(int fw = 0; fw < filter_w; ++fw){
                                        sum[i][j] += bufX[i+fh][j+fw] * bufF[fh][fw];
                                    }
                                }
                            }
                        }
                    }
                    for(int i = 0; i < min_y; i++){
                        for(int j = 0; j < min_x; j++){
                            if( (oh+i)%stride_h == 0 && (ow+j)%stride_w == 0) //TODO to be optimized
                              y_data[n * out_channels * o_h * o_w + oc * o_h * o_w + (oh+i)/stride_h * o_w + (ow+j)/stride_w] = sum[i][j] + tmpb;
                        }
                    }
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
    int i_n = 1;
    int i_c = 1;
    int i_h = 1;
    int i_w = 1;
    int f_h = 5;
    int f_w = 5;
    int o_c = 2;
    int padding_h = 1;
    int padding_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w= 1;

    for(i_n = 1; i_n <= 256; i_n++){
    for(i_c = 1; i_c <= 256; i_c++){
    for(i_h =256; i_h <= 256; i_h++){
        i_w = i_h;
    for(f_h = 7; f_h <=11; f_h+=2){
        f_w = f_h;
    for(o_c = 32; o_c <= 512; o_c+=32){
        int tmp_f_h = (f_h - 1) * dilation_h + 1;
        int tmp_f_w = (f_w - 1) * dilation_w + 1;
        int o_h = (i_h + 2 * padding_h - tmp_f_h) / stride_h + 1;
        int o_w = (i_w + 2 * padding_w - tmp_f_w) / stride_w + 1;
        if(o_h <= 0 || o_w <= 0) continue;
        std::cout << i_n << " " << i_c << " " << i_h << " " << i_w << " " << f_h << " " << f_w << " " << o_c << std::endl;
        size_t s_i = i_n * i_c * i_h * i_w;
        size_t s_f = o_c * i_c * f_h * f_w;
        size_t s_o = i_n * o_c * o_h * o_w;
        int *input = new int[s_i];
        int *filter = new int[s_f];
        int *b_data = new int[o_c];
        int *output = new int[s_o];
        for(int i = 0; i < s_i; i++)
            input[i] = 1;
        for(int i = 0; i < s_f; i++)
            filter[i] = 1;
        for(int i = 0; i < o_c; i++)
            b_data[i] = 1;
    //    print(input, i_c, i_h, i_w);
        clock_t start = clock();
/*        for(int i = 0; i < 1; i++){
            conv_cpu(input, i_n, i_h, i_w, i_c,
                    filter, f_h, f_w,
                    b_data,
                    output, o_h, o_w, o_c,
                    stride_h, stride_w,
                    padding_h, padding_w,
                    dilation_h, dilation_w);
        }*/
        clock_t end = clock();
    //    print(output, i_n, o_c, o_h, o_w);
//        std::cout << "cpu time: " << end-start << std::endl;
    //    int *output3 = new int[s_o];
    //    clock_t start2 = clock();
    //    for(int i = 0; i < 10; i++){
    //        conv_cpu_v2(input, i_n, i_h, i_w, i_c,
    //                filter, f_h, f_w,
    //                b_data,
    //                output3, o_h, o_w, o_c,
    //                stride, stride,
    //                padding);
    //    }
    //    clock_t end2 = clock();
    ////  print(output3, i_n, o_c, o_h, o_w);
    //    std::cout << "cpu time: " << end2-start2 << std::endl;
    //    int ret = memcmp(output, output3, s_o*sizeof(int32_t));
    //    std::cout << (ret == 0 ? "pass" : "failed") << std::endl;

        int* output2 = new int[s_o];
        const char* errorStr = cuda_conv2d(
            input, i_n, i_c, i_h, i_w,
            filter, o_c, i_c, f_h, f_w,
            b_data,
            padding_h, padding_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            1,
            output2, i_n, o_c, o_h, o_w, 0, true);

        clock_t gpu_end = clock();
        std::cout << "gpu all time: " << gpu_end - end << std::endl;
        if(errorStr != NULL){
            std::cout << errorStr << std::endl;
            return 0;
        }
    //    print(output2, i_n, o_c, o_h, o_w);
        int ret2 = memcmp(output, output2, sizeof(int) * s_o);
 /*       std::cout << (ret2 == 0 ? "pass" : "failed") << std::endl;
        if(ret2 != 0){
            std::cout << "cpu output:\n";
            print(output, i_n, i_c, i_h, i_w);
            std::cout << "cuda output:\n";
            print(output2, i_n, o_c, o_h, o_w);
            return 0;
        }*/
        delete input;
        delete filter;
        delete b_data;
        delete output;
        delete output2;
    }
    }
    }
    }
    }
    return 0;
}
