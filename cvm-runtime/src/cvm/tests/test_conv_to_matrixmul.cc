#include <iostream>
#include <string.h>

using namespace std;

void matrix_mul(const int32_t *a, const int32_t *b, const int32_t *bias,
        int32_t *c, const int M, const int K, const int N){
    memset(c, 0, sizeof(int32_t) * N * M);
    cout << "m=" << M << ", k=" << K << ", n=" << N << endl;
    for(int i = 0; i < M; i++){
        for(int k = 0; k < K; k+=4){
            register int32_t aV[4] = {0};
            aV[0] = a[i*K + k + 0];
            aV[1] = k+1 < K ? a[i*K + k + 1] : 0;
            aV[2] = k+2 < K ? a[i*K + k + 2] : 0;
            aV[3] = k+3 < K ? a[i*K + k + 3] : 0;
            for(int j = 0; j < N; j++){
                register int tc = c[i*N+j];
                tc += aV[0] * b[(k + 0) * N + j];
                tc += k+1 < K ? aV[1] * b[(k + 1) * N + j] : 0;
                tc += k+2 < K ? aV[2] * b[(k + 2) * N + j] : 0;
                tc += k+3 < K ? aV[3] * b[(k + 3) * N + j] : 0;
                c[i*N + j] = tc;
            }
        }
    }
    for(int i = 0; i < M; i++){
        int biasV = bias[i];
        for(int j = 0; j < N; j++){
            c[i*N+j] += biasV;
        }
    }
}
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}
void im2col_cpu(const int32_t* data_im, const int channels,
        const int height, const int width, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        int32_t* data_col) {
    const int output_h = (height + 2 * pad_h -
            (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -
            (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_row = -pad_h + kernel_row * dilation_h;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    } else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                *(data_col++) = data_im[input_row * width + input_col];
                            } else {
                                *(data_col++) = 0;
                            }
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}

void conv_cpu(int* x_data, int n_batch, int x_h, int x_w, int in_channels,
        int *w_data, int filter_h, int filter_w,
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
int main(){
    int i_n = 1;
    int i_c = 1;
    int i_h = 32;
    int i_w = 32;
    int f_h = 3;
    int f_w = 3;
    int o_c = 1024;
    int padding_h = 0;
    int padding_w = 0;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w= 1;

    for(i_n = 1; i_n <= 1; i_n++){
    for(i_c = 1024; i_c <= 1024; i_c++){
    for(i_h = 32; i_h <= 32; i_h++){
        i_w = i_h;
    for(f_h = 1; f_h <= 1; f_h+=2){
        f_w = f_h;
    for(o_c = 1024; o_c <= 1024; o_c++){
        int tmp_f_h = (f_h - 1) * dilation_h + 1;
        int tmp_f_w = (f_w - 1) * dilation_w + 1;
        int o_h = (i_h + 2 * padding_h - tmp_f_h) / stride_h + 1;
        int o_w = (i_w + 2 * padding_w - tmp_f_w) / stride_w + 1;
//        if(o_h <= 0 || o_w <= 0) continue;
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
        for(int i = 0; i < 1; i++){
            conv_cpu(input, i_n, i_h, i_w, i_c,
                    filter, f_h, f_w,
                    b_data,
                    output, o_h, o_w, o_c,
                    stride_h, stride_w,
                    padding_h, padding_w,
                    dilation_h, dilation_w);
        }
        clock_t end = clock();
        cout << "conv : " << (end-start)*1.0 / CLOCKS_PER_SEC << endl;
        int32_t *output2 = new int[s_o];
        clock_t im2col_start = clock();
        int32_t *data_col = new int[i_c*f_h*f_w * o_h * o_w];
        for(int i = 0; i < i_n; i++){
            im2col_cpu(input, i_c, i_h, i_w, f_h, f_w, padding_h, padding_w, stride_h, stride_w,
                    dilation_h, dilation_w, data_col);
            matrix_mul(filter, data_col, b_data, output2 + i * o_c * o_h * o_w, o_c, i_c * f_h * f_w , o_h * o_w);
        }
        clock_t im2col_end = clock();
        cout << "im2col conv : " << (im2col_end-im2col_start)*1.0 / CLOCKS_PER_SEC << endl;

        int ret = memcmp(output, output2, sizeof(int32_t) * s_o);
        cout << (ret == 0 ? "success" : "failed") << std::endl;
        if(ret != 0) return 0;
        delete input;
        delete filter;
        delete b_data;
        delete output;
        delete output2;
        delete data_col;
    }
    }
    }
    }
    }
    return 0;
}
