#include <cvm/c_api.h>
#include <cvm/model.h>
#include <iostream>
#include <thread>
#include <omp.h>
using namespace std;
int run_LIF(string model_root) {
    string json_path = model_root + "/symbol";
    string params_path = model_root + "/params";
    cerr << "load " << json_path << "\n";
    cerr << "load " << params_path << "\n";
    cvm::runtime::CVMModel* model = static_cast<cvm::runtime::CVMModel*>(
            CVMAPILoadModel(json_path.c_str(), params_path.c_str(), 0, 1)
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
    int n_run = 10;
    for (int i = 0; i < n_run; i++) {
        if (i % 10 == 0)
                cerr << "i = " << i << "\n";
        CVMAPIInfer(model, input.data(), output.data());
    }
    double ellapsed_time = (omp_get_wtime() - start) / n_run;
    cout << "total gemm.trans time:" << cvm::runtime::transpose_int8_avx256_transpose_cnt / n_run << "\n";
    cout << "total  gemm.gemm time:" << cvm::runtime::transpose_int8_avx256_gemm_cnt / n_run << "\n";
    cout << "total     im2col time:" << cvm::runtime::im2col_cnt / n_run<< "\n";
    double sum_time = 0;
    sum_time +=  cvm::runtime::transpose_int8_avx256_transpose_cnt / n_run;
    sum_time +=  cvm::runtime::transpose_int8_avx256_gemm_cnt / n_run;
    sum_time +=  cvm::runtime::im2col_cnt / n_run;
    cout << "total gemm time" << (sum_time) << "/" << ellapsed_time
         << " " <<  sum_time / ellapsed_time <<"\n";
    sum_time = 0;
    cout << "total navive dense time" << (cvm::runtime::cvm_op_dense_cnt) << "/" << ellapsed_time
         << " " <<  cvm::runtime::cvm_op_dense_cnt / ellapsed_time <<"\n";
    sum_time = (cvm::runtime::cvm_op_maxpool_cnt) / n_run;
    cout << "total maxpool time" <<  sum_time << "/" << ellapsed_time
         << " " <<  sum_time / ellapsed_time <<"\n";
    sum_time = (cvm::runtime::cvm_op_broadcast_cnt) / n_run;
    cout << "total broadcast time" <<  sum_time << "/" << ellapsed_time
         << " " <<  sum_time / ellapsed_time <<"\n";


    sum_time +=  cvm::runtime::cvm_op_clip_cnt / n_run;
    cout << "total clip time" << (sum_time) << "/" << ellapsed_time
         << " " <<  sum_time / ellapsed_time <<"\n";


    sum_time +=  cvm::runtime::cvm_op_rightshift_cnt / n_run;
    cout << "total rightshift time" << (sum_time) << "/" << ellapsed_time
         << " " <<  sum_time / ellapsed_time <<"\n";

    sum_time +=  cvm::runtime::cvm_op_concat_cnt / n_run;
    cout << "total concat time" << (sum_time) << "/" << ellapsed_time
         << " " <<  sum_time / ellapsed_time <<"\n";

    CVMAPIFreeModel(model);
    return 0;
}
int main() {
    vector<std::thread> threads;
    for (int t = 0; t < 1; ++t) {
        cerr << "threads t = " << t << "\n";
        threads.push_back(thread([&]() {
                string model_root = "/home/tian/model_storage/resnet50_v1/data/";
                // model_root = "/home/tian/cortex_fullnode_storage/cifar_resnet20_v2/data";
                // model_root = "/home/lizhen/storage/mnist/data/";
                // model_root = "/home/lizhen/storage/animal10/data";
                model_root = "/home/tian/cortex_fullnode_storage/imagenet_inceptionV3/data";
                run_LIF(model_root);
                //run_LIF(model_root);
        }));
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return 0;
}
