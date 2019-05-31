#include <cvm/c_api.h>
#include <cvm/model.h>
#include <iostream>
#include <thread>
#include <omp.h>
using namespace std;
int run_LIF(string model_root) {

    cvm::runtime::transpose_int8_avx256_transpose_cnt = 0;
    cvm::runtime::transpose_int8_avx256_gemm_cnt = 0;
    cvm::runtime::im2col_cnt = 0;
    cvm::runtime::cvm_op_rightshift_cnt = 0;
    cvm::runtime::cvm_op_clip_cnt = 0;
    cvm::runtime::cvm_op_dense_cnt = 0;
    cvm::runtime::cvm_op_maxpool_cnt = 0;
    cvm::runtime::cvm_op_broadcast_cnt = 0;
    cvm::runtime::cvm_op_concat_cnt = 0;
    cvm::runtime::cvm_op_upsampling_cnt = 0;
    cvm::runtime::cvm_op_inline_matmul_cnt = 0;

    string json_path = model_root + "/symbol";
    string params_path = model_root + "/params";
    cerr << "load " << json_path << "\n";
    cerr << "load " << params_path << "\n";
    cvm::runtime::CVMModel* model = static_cast<cvm::runtime::CVMModel*>(
        CVMAPILoadModel(json_path.c_str(), params_path.c_str(), 0, 0)
    );
    if (model == nullptr) {
        std::cerr << "model loaded failed\n";
        return -1;
    }
    cerr << "ops " << CVMAPIGetGasFromModel(model) / 1024 / 1024 << "\n";
    vector<char> input, output;
    int input_size = CVMAPIGetInputLength(model);
    int output_size = CVMAPIGetOutputLength(model);
    input.resize(input_size); // 1 * 1 * 28 * 28);
    output.resize(output_size); //1 * 10);
    double start = omp_get_wtime();
    int n_run = 1;
    for (int i = 0; i < n_run; i++) {
        if (i % 10 == 0)
                cerr << "i = " << i << "\n";
        CVMAPIInfer(model, input.data(), output.data());
    }
    double ellapsed_time = (omp_get_wtime() - start) / n_run;
    cout << "total gemm.trans time: " << cvm::runtime::transpose_int8_avx256_transpose_cnt / n_run << "\n";
    cout << "total  gemm.gemm time: " << cvm::runtime::transpose_int8_avx256_gemm_cnt / n_run << "\n";
    cout << "total     im2col time: " << cvm::runtime::im2col_cnt / n_run<< "\n";
    double sum_time = 0;
    sum_time +=  cvm::runtime::transpose_int8_avx256_transpose_cnt / n_run;
    sum_time +=  cvm::runtime::transpose_int8_avx256_gemm_cnt / n_run;
    sum_time +=  cvm::runtime::im2col_cnt / n_run;
    cout << "total       gemm time: " << (sum_time) << "/" << ellapsed_time
         << " " <<  sum_time / ellapsed_time <<"\n";
    sum_time = cvm::runtime::cvm_op_dense_cnt / n_run;
    cout << "total naivedense time: " << (sum_time) << "/" << ellapsed_time
         << " " <<  sum_time / ellapsed_time <<"\n";
    sum_time = (cvm::runtime::cvm_op_maxpool_cnt) / n_run;
    cout << "total    maxpool time: " <<  sum_time << "/" << ellapsed_time
         << " " <<  sum_time / ellapsed_time <<"\n";
    sum_time = (cvm::runtime::cvm_op_broadcast_cnt) / n_run;
    cout << "total  broadcast time: " <<  sum_time << "/" << ellapsed_time
         << " " <<  sum_time / ellapsed_time <<"\n";


    sum_time =  cvm::runtime::cvm_op_clip_cnt / n_run;
    cout << "total       clip time: " << (sum_time) << "/" << ellapsed_time
         << " " <<  sum_time / ellapsed_time <<"\n";


    sum_time =  cvm::runtime::cvm_op_rightshift_cnt / n_run;
    cout << "total rightshift time: " << (sum_time) << "/" << ellapsed_time
         << " " <<  sum_time / ellapsed_time <<"\n";

    sum_time =  cvm::runtime::cvm_op_concat_cnt / n_run;
    cout << "total    concat time: " << (sum_time) << "/" << ellapsed_time
         << " " <<  sum_time / ellapsed_time <<"\n";

    sum_time =  cvm::runtime::cvm_op_upsampling_cnt / n_run;
    cout << "total upsampling time: " << (sum_time) << "/" << ellapsed_time
         << " " <<  sum_time / ellapsed_time <<"\n";

    sum_time =  cvm::runtime::cvm_op_inline_matmul_cnt / n_run;
    cout << "total matmul     time: " << (sum_time) << "/" << ellapsed_time
         << " " <<  sum_time / ellapsed_time <<"\n";
    CVMAPIFreeModel(model);
    return 0;
}
void test_thread() {
    vector<std::thread> threads;
    for (int t = 0; t < 1; ++t) {
        cerr << "threads t = " << t << "\n";
        threads.push_back(thread([&]() {
                string model_root = "/home/tian/model_storage/resnet50_v1/data/";
                // model_root = "/home/tian/cortex_fullnode_storage/cifar_resnet20_v2/data";
                // model_root = "/home/lizhen/storage/mnist/data/";
                // model_root = "/home/lizhen/storage/animal10/data";
                // model_root = "/home/tian/cortex_fullnode_storage/imagenet_inceptionV3/data";
                run_LIF(model_root);
                //run_LIF(model_root);
        }));
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

void test_models() {
    auto model_roots = {
        // "/home/tian/model_storage/resnet50_v1/data/",
        // "/home/tian/cortex_fullnode_storage/imagenet_inceptionV3/data",
        // "/home/tian/model_storage/animal10/data",
        // "/home/tian/model_storage/mnist/data",
        // "/home/tian/model_storage/resnet50_v2/data",
        // "/home/tian/model_storage/vgg16_gcv/data",
        // "/home/tian/model_storage/vgg19_gcv/data",
        // "/home/tian/model_storage/squeezenet_gcv1.1/data",
        // "/home/tian/model_storage/squeezenet_gcv1.0/data",
        // "/home/tian/model_storage/octconv_resnet26_0.250/data"
        "/home/tian/model_storage/yolo3_darknet53/data"
    };
    for (auto model_root : model_roots) {
        run_LIF(model_root);
    }
}
int main() {
    test_models();
    return 0;
}
