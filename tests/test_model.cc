#include <cvm/c_api.h>
#include <cvm/model.h>
#include <iostream>
#include <thread>
#include <omp.h>
#include <cvm/runtime/registry.h>
#include <cvm/op.h>
#include "npy.hpp"
using namespace std;

using cvm::runtime::PackedFunc;
using cvm::runtime::Registry;

#ifndef USE_GPU
#define USE_GPU  0
#endif

#define CHECK_STATUS(x, msg) \
  if (x != SUCCEED) { \
    cerr << "STATUS ERROR: " << x << " " << msg << "\n"; \
    return -1; \
  }

void read_data(const char *filename, vector<unsigned long> &shape, vector<int32_t>& data){
    FILE *fp = fopen(filename, "r");
    if(fp == NULL){
        return;
    }
    int32_t shape_dim = 0;
    fscanf(fp, "%d ", &shape_dim);
    printf("shape_dim = %d\n", shape_dim);
    shape.resize(shape_dim);
    uint64_t size = 1;
    for(int i = 0; i < shape_dim; i++){
        int64_t value = 0;
        fscanf(fp, "%ld ", &value);
        shape[i] = value;
        size *= shape[i];
    }
    data.resize(size);
    for(int i = 0; i < size; i++){
        int32_t value = 0;
        fscanf(fp, "%d ", &value);
        data[i] = value;
    }
    fclose(fp);
}

void write_result(const char *filename, vector<char>& data){
    FILE* fp = fopen(filename, "w");
    if(fp == NULL){
        printf("open file %s failed\n", filename);
        return;
    }
    fprintf(fp, "%zul\n", data.size());
    for(int i = 0; i < data.size(); i++){
        fprintf(fp, "%d ", data[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
}

void compare_result(const char *filename, vector<char>& data){
    FILE* fp = fopen(filename, "r");
    if(fp == NULL){
        printf("open file %s failed\n", filename);
        return;
    }
    int n = 0;
    fscanf(fp, "%d", &n);
    assert(n == data.size());

    for(int i = 0; i < data.size(); i++){
      int value;
      fscanf(fp, "%d ", &value);
      assert((int)data[i] == value);
    }
    fclose(fp);
    printf("compare result: success\n\n");
}

struct OpArgs {
  std::vector<DLTensor> args;
  std::vector<CVMValue> arg_values;
  std::vector<int> arg_tcodes;
  std::vector<int64_t> shape_data;
};

int run_LIF(string model_root, int device_type = 0) {
#if(USE_GPU==0)
  cvm::runtime::transpose_int8_avx256_transpose_cnt = 0;
  cvm::runtime::transpose_int8_avx256_gemm_cnt = 0;
  cvm::runtime::im2col_cnt = 0;
  cvm::runtime::cvm_op_cvm_shift_cnt = 0;
  cvm::runtime::cvm_op_clip_cnt = 0;
  cvm::runtime::cvm_op_dense_cnt = 0;
  cvm::runtime::cvm_op_maxpool_cnt = 0;
  cvm::runtime::cvm_op_broadcast_cnt = 0;
  cvm::runtime::cvm_op_concat_cnt = 0;
  cvm::runtime::cvm_op_upsampling_cnt = 0;
  cvm::runtime::cvm_op_inline_matmul_cnt = 0;
  cvm::runtime::cvm_op_chnwise_conv_cnt = 0;
  cvm::runtime::cvm_op_depthwise_conv_cnt = 0;
  cvm::runtime::cvm_op_chnwise_conv1x1_cnt = 0;
#endif

  string json_path = model_root + "/symbol";
  string params_path = model_root + "/params";
  cerr << "load " << json_path << "\n";
  cerr << "load " << params_path << "\n";
  std::string params, json;
  {
    std::ifstream input_stream(json_path, std::ios::binary);
    json = string((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());
    input_stream.close();
  }
  {
    std::ifstream input_stream(params_path, std::ios::binary);
    params  = string((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());
    input_stream.close();
  }
  void *net;
  auto status = CVMAPILoadModel(json.c_str(), json.size(),
                                params.c_str(), params.size(),
                                &net,
                                device_type, 0);
  cerr << "model loaded\n";
  CHECK_STATUS(status, "model loaded failed");

  unsigned long long gas = 0;
  status = CVMAPIGetGasFromModel(net, &gas);
  CHECK_STATUS(status, "gas invalid");
  cerr << "ops " << gas / 1024 / 1024 << "\n";
  // API only accepts byte array
  vector<char> input, output;
  unsigned long long input_size, output_size;
  CVMAPIGetInputLength(net, &input_size);
  CVMAPIGetOutputLength(net, &output_size);
  input.resize(input_size, 0); // 1 * 1 * 28 * 28);
  output.resize(output_size, 0); //1 * 10);
  if (model_root.find("trec") != string::npos)
  {
    vector<int32_t> input_int32_t;
    std::vector<unsigned long> tshape;
    npy::LoadArrayFromNumpy("/data/std_out/trec/data.npy", tshape, input_int32_t);
    std::cerr << "Loading a int32 data and cast to byte array: "
              << input.size() << " " << input_int32_t.size() << "\n";
    memcpy(input.data(), input_int32_t.data(), input.size());
  }
  else if (model_root.find("yolo") != string::npos)
  {
    std::vector<unsigned long> tshape;
    npy::LoadArrayFromNumpy("/tmp/yolo/out/data.npy", tshape, input);
    std::cerr << tshape.size() << "\n";
    for (auto x : tshape) {
      std::cerr << x << " ";
    }
    std::cerr << "\n";
  }
  else if (model_root.find("std_out") != string::npos)
  {
    string data_file = model_root + "/data.npy";
    std::vector<unsigned long> tshape;
    npy::LoadArrayFromNumpy(data_file, tshape, input);
    std::cerr << tshape.size() << "\n";
    for (auto x : tshape) {
      std::cerr << x << " ";
    }
    std::cerr << "\n";
  }
  else if (model_root.find("3145ad19228c1cd2d051314e72f26c1ce77b7f02") != string::npos)
  {
    string data_file =  model_root + "/cpu.txt";
    std::vector<unsigned long> tshape;
    std::vector<int32_t> data;
    //npy::LoadArrayFromNumpy(data_file, tshape, input);
    read_data(data_file.c_str(), tshape, data);
    std::cerr << tshape.size() << "\n";
    for (int i = 0; i < data.size(); i++) {
      input[i]= (int8_t)data[i];
      if(i < 10){
        printf("%d ", input[i]);
      }
    }
    printf("\n");
  }

  double start = omp_get_wtime();
  int n_run = 1;
  for (int i = 0; i < n_run; i++) {
    if (i % 10 == 0)
      cerr << "i = " << i << "\n";
    status = CVMAPIInference(net, input.data(), input.size(), output.data());
    CHECK_STATUS(status, "inference failed");
  }
  status = CVMAPIFreeModel(net);
  CHECK_STATUS(status, "free model failed");
#if(USE_GPU == 0)
  double ellapsed_time = (omp_get_wtime() - start) / n_run;
  cout << "total time : " << ellapsed_time / n_run << "\n";
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


  sum_time =  cvm::runtime::cvm_op_cvm_shift_cnt / n_run;
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

  sum_time =  cvm::runtime::cvm_op_elemwise_cnt / n_run;
  cout << "total elemwise   time: " << (sum_time) << "/" << ellapsed_time
    << " " <<  sum_time / ellapsed_time <<"\n";

  sum_time =  cvm::runtime::cvm_op_chnwise_conv_cnt / n_run;
  cout << "total chn conv2d time: " << (sum_time) << "/" << ellapsed_time
    << " " <<  sum_time / ellapsed_time <<"\n";

  sum_time =  cvm::runtime::cvm_op_depthwise_conv_cnt / n_run;
  cout << "total depth conv2d time: " << (sum_time) << "/" << ellapsed_time
    << " " <<  sum_time / ellapsed_time <<"\n";

  sum_time =  cvm::runtime::cvm_op_chnwise_conv1x1_cnt / n_run;
  cout << "total chnconv2d1x1 time: " << (sum_time) << "/" << ellapsed_time
    << " " <<  sum_time / ellapsed_time <<"\n";
#endif

  if (json_path.find("yolo") != string::npos) {
    uint64_t n_bytes = 4;
    uint64_t ns =  output.size() / n_bytes;
    std::cout << "yolo output size = " << ns << " n_bytes = " << n_bytes << "\n";
    int32_t* int32_output = static_cast<int32_t*>((void*)output.data());
    for (auto i = 0; i < std::min(60UL, ns); i++) {
      std::cout << (int32_t)int32_output[i] << " ";
      if ((i + 1) % 6 == 0)
        std::cout << "\n";
    }
    // last 60 rows of results
    if (ns > 60) {
      for (auto i = (size_t)(std::max(0, ((int)(ns) - 60))); i < ns; i++) {
        std::cout << (int32_t)int32_output[i] << " ";
        if ((i + 1) % 6 == 0)
          std::cout << "\n";
      }
    }
    std::cout << "\n";
  } else {
    std::cout << "output size = " << output.size() << "\n";
    for (auto i = 0; i < std::min(6UL * 10, output.size()); i++) {
      std::cout << (int32_t)output[i] << " ";
    }
    std::cout << "\n";
    if (output.size() > 60) {
      for (auto i = (size_t)(std::max(0, ((int)(output.size()) - 6 * 10))); i < output.size(); i++) {
        std::cout << (int32_t)output[i] << " ";
      }
      std::cout << "\n";
    }

    string out_file = model_root + "/result_0.txt";
   // write_result(out_file.c_str(), output);
   compare_result(out_file.c_str(), output);
   // string data_file = model_root + "/result_0.npy";
   // vector<unsigned long> tshape;
   // vector<int32_t> tout;
   // npy::LoadArrayFromNumpy(data_file, tshape, tout);
   // cout << tout.size() << " " << output.size() << endl;
   // for(int i = 0; i < tout.size() && i < 60; i++){
   //   cout << tout[i] << " ";
   // }
   // cout << endl;
   // for(int i = 0; i < tout.size(); i++){
   //     if((int32_t)output[i] != tout[i]){
   //        cout << "failed!!!!! : " << i << " " << (int32_t)output[i] << " " << (int32_t)tout[i] << endl;
   //     }
   //     assert((int32_t)output[i] == tout[i]);
   // }
  }
  return 0;
}
void test_thread() {
  vector<std::thread> threads;
  for (int t = 0; t < 1; ++t) {
    cerr << "threads t = " << t << "\n";
    threads.push_back(thread([&]() {
          string model_root = "/home/tian/model_storage/resnet50_v1/data/";
          // model_root = "/home/kaihuo/cortex_fullnode_storage/cifar_resnet20_v2/data";
          // model_root = "/home/tian/storage/mnist/data/";
          // model_root = "/home/tian/storage/animal10/data";
          // model_root = "/home/kaihuo/cortex_fullnode_storage/imagenet_inceptionV3/data";
          run_LIF(model_root);
          //run_LIF(model_root);
          }));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

int test_models(int device_type = 0) {
  auto model_roots = {
    "/data/std_out/null",
    "/data/std_out/resnet50_mxg",
    "/data/std_out/resnet50_v2",
    "/data/std_out/qd10_resnet20_v2",
    "/data/std_out/trec",
    "/data/new_cvm/yolo3_darknet53_voc/data",
    "/data/lz_model_storage/dcnet_mnist_v1/data",
    "/data/lz_model_storage/mobilenetv1.0_imagenet/data",
    "/data/lz_model_storage/resnet50_v1_imagenet/data",
    "/data/lz_model_storage/animal10/data",
    "/data/lz_model_storage/resnet50_v2/data",
    "/data/lz_model_storage/vgg16_gcv/data",
    "/data/lz_model_storage/sentiment_trec/data",
    "/data/lz_model_storage/vgg19_gcv/data",
    "/data/lz_model_storage/squeezenet_gcv1.1/data",
    "/data/lz_model_storage/squeezenet_gcv1.0/data",
    // invalid has strange attribute in operator elemwise_add.
    //"/data/lz_model_storage/octconv_resnet26_0.250/data",
    "/data/std_out/resnet50_mxg/",
    "/data/std_out/resnet50_v2",
    "/data/std_out/qd10_resnet20_v2",
    "/data/std_out/random_3_0/",
    "/data/std_out/random_3_1/",
    "/data/std_out/random_3_2/",
    "/data/std_out/random_3_3/",
    "/data/std_out/random_3_4/",
    "/data/std_out/random_3_5/",
    "/data/std_out/random_4_0/",
    "/data/std_out/random_4_1/",
    // "/data/std_out/random_4_2/",
    // "/data/std_out/random_4_3/",
    // "/data/std_out/random_4_4/",
    "/data/std_out/random_4_5/",
    "/data/std_out/random_4_6/",
    "/data/std_out/random_4_7/",
    "/data/std_out/random_4_8/",
    "/data/std_out/random_4_9/",
    "/data/std_out/log2",
    "./tests/3145ad19228c1cd2d051314e72f26c1ce77b7f02/",
    "/data/std_out/lr_attr",
  };
  for (auto model_root : model_roots) {
    auto ret = run_LIF(model_root, device_type);
    if (ret == -1) return -1;
  }
  return 0;
}
int main() {
 if (test_models(USE_GPU) != 0)
   return -1;
  return 0;
}
