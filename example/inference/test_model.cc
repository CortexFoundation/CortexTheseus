#include <cvm/c_api.h>
#include <cvm/model.h>
#include <iostream>
#include <thread>
using namespace std;
int run_LIF(string model_root) {
    string json_path = model_root + "/symbol";
    string params_path = model_root + "/params";
    cerr << "load " << json_path << "\n";
    cerr << "load " << params_path << "\n";
    cvm::runtime::CVMModel* model = static_cast<cvm::runtime::CVMModel*>(
            CVMAPILoadModel(json_path.c_str(), params_path.c_str())
    );
    if (model == nullptr) {
        std::cerr << "model loaded failed\n";
        return -1;
    }
    vector<char> input, output;
    int input_size = CVMAPIGetInputLength(model);
    int output_size = CVMAPIGetOutputLength(model);
    input.resize(input_size); // 1 * 1 * 28 * 28);
    output.resize(output_size); //1 * 10);
    CVMAPIInfer(model, input.data(), output.data());
    CVMAPIFreeModel(model);
    return 0;
}
int main() {
    vector<std::thread> threads;
    for (int t = 0; t < 1; ++t) {
        cerr << "threads t = " << t << "\n";
        threads.push_back(thread([&]() {
            for (int i = 0; i < 10; i++) {
                if (i % 10 == 0)
                    cerr << "i = " << i << "\n";
                string model_root = "/home/tian/cortex_fullnode_storage/imagenet_inceptionV3/data";
                run_LIF(model_root);
                //model_root = "/home/tian/cortex_fullnode_storage/cifar_resnet20_v2/data";
                //run_LIF(model_root);
            }
        }));
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return 0;
}
