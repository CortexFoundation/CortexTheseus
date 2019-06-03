#include <cvm/c_api.h>
#include <cvm/dlpack.h>
#include <cvm/runtime/module.h>
#include <cvm/runtime/registry.h>
#include <cvm/runtime/packed_func.h>
#include <cvm/model.h>

#include <fstream>
#include <iterator>
#include <algorithm>
#include <stdio.h>
#include <string.h>

#include <time.h>
#include "npy.hpp"
#include <omp.h>

int dtype_code = kDLInt;
int dtype_bits = 8;
int dtype_lanes = 1;
int device_type = kDLCPU;
int device_id = 0;

int verify(DLTensor *output, std::string filename){
    std::vector<unsigned long> tshape;
    std::vector<int32_t> tout;
    npy::LoadArrayFromNumpy(filename, tshape, tout);
    int ret = std::memcmp(static_cast<int32_t*>(output->data), tout.data(), sizeof(int32_t) * tout.size());
    if(ret != 0){
        for(int i = 0; i < 10; i++){
            printf("%d ", tout[i]);
        }
        printf("\n");
        for(int i = 0; i < 10; i++){
            printf("%d ", static_cast<int32_t*>(output->data)[i]);
        }
        printf("\n");
    }
    return ret;
}
long Run(const std::string& graph_str, const std::string& param_str, int device_type, int device_id,
        char *input_data, int32_t* output_data) {
    cvm::runtime::CVMModel* model = new cvm::runtime::CVMModel(graph_str, DLContext{static_cast<DLDeviceType>(device_type), device_id});
    model->LoadParams(param_str);
    int input_length = model->GetInputLength();
    int output_length = model->GetOutputLength();

    int ret = 0;
    if (input_data == nullptr) {
        std::cerr << "input_data error" << std::endl;
        ret = -1;
    } else if (output_data == nullptr) {
        std::cerr << "output error" << std::endl;
        ret = -1;
    } else {
        DLTensor* input = model->PlanInput(input_data);
        auto outputs = model->PlanOutput();
        if (input == nullptr) {
            std::cerr << "input == nullptr || output == nullptr" << std::endl;
            ret = -1;
        } else {
            double start = omp_get_wtime();
            ret = model->Run(input, outputs);
            double end = omp_get_wtime();

            std::cout << "run time : " << end - start << " s" << std::endl;
            std::cout << "verify result 0 " << (verify(outputs[0], "/tmp/yolo/out/result_0.npy") == 0 ? "success\n" : "failed\n");
            std::cout << "verify result 1 " << (verify(outputs[1], "/tmp/yolo/out/result_1.npy") == 0 ? "success\n" : "failed\n");
            std::cout << "verify result 2 " << (verify(outputs[2], "/tmp/yolo/out/result_2.npy") == 0 ? "success\n" : "failed\n");

            if (input)
                CVMArrayFree(input);
            for (int i = 0; i < outputs.size(); ++i)
                CVMArrayFree(outputs[i]);
        }
    }

    return 0;
}

int main()
{
    std::vector<unsigned long> tshape;
    std::vector<char> tdata;
    npy::LoadArrayFromNumpy("/tmp/yolo/out/data.npy", tshape, tdata);

    clock_t read_t1 = clock();
    // parameters in binary
    std::ifstream params_in("/tmp/yolo/yolo3_darknet53_voc.all.nnvm.compile.params", std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();

    // parameters need to be CVMByteArray type to indicate the binary data
    std::ifstream json_in2("/tmp/yolo/yolo3_darknet53_voc.all.nnvm.compile.json", std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in2)), std::istreambuf_iterator<char>());
    json_in2.close();
    // json graph
    std::cout << "loadfromfile time " << (clock() - read_t1) * 1000 / CLOCKS_PER_SEC  << "s"<< std::endl;

    int32_t *output = new int32_t[1*100];
    for (int i = 0; i < 1; i++) {
        Run(json_data, params_data, (int)kDLCPU, 0, tdata.data(), output);
    }

    delete output;
    return 0;
}
