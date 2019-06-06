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
struct OpArgs {
  std::vector<DLTensor> args;
  std::vector<CVMValue> arg_values;
  std::vector<int> arg_tcodes;
  std::vector<int64_t> shape_data;
};

void test_op_take() {
  CVMValue t_attr;
  const PackedFunc* op = Registry::Get("cvm.runtime.cvm.take");
  cvm::NodeAttrs* attr;
  std::shared_ptr<OpArgs> arg_ptr = std::make_shared<OpArgs>();
  t_attr.v_handle = (void*)attr;
  arg_ptr->arg_values.push_back(t_attr);
  arg_ptr->arg_tcodes.push_back(kHandle);
  cvm::runtime::CVMRetValue rv;
  cvm::runtime::CVMArgs targs(
      arg_ptr->arg_values.data(),
      arg_ptr->arg_tcodes.data(),
      static_cast<int>(arg_ptr->arg_values.size())
      );
  // (*op)(ta);

}

int run_LIF(string model_root) {

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
  input.resize(input_size, 0); // 1 * 1 * 28 * 28);
  output.resize(output_size, 0); //1 * 10);
  if (model_root.find("trec") != string::npos)
  {
    vector<int32_t> input_int32_t;
    std::vector<unsigned long> tshape;
    npy::LoadArrayFromNumpy("/tmp/trec/out/data.npy", tshape, input_int32_t);
    std::cerr << "Loading a int32 data and cast to byte array: "
              << input.size() << " " << input_int32_t.size() << "\n";
    memcpy(input.data(), input_int32_t.data(), input.size());
  }
  if (model_root.find("yolo") != string::npos)
  {
    std::vector<unsigned long> tshape;
    npy::LoadArrayFromNumpy("/tmp/yolo/out/data.npy", tshape, input);
    std::cerr << tshape.size() << "\n";
    for (auto x : tshape) {
      std::cerr << x << " ";
    }
    std::cerr << "\n";
  }
  double start = omp_get_wtime();
  int n_run = 1;
  for (int i = 0; i < n_run; i++) {
    if (i % 10 == 0)
      cerr << "i = " << i << "\n";
    CVMAPIInfer(model, input.data(), output.data());
  }
  CVMAPIFreeModel(model);
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

  if (model_root.find("yolo") != string::npos) {
    uint64_t ns =  output.size() / 4 / 4;
    std::cout << "yolo output size = " << ns << "\n";
    int32_t* int32_output = static_cast<int32_t*>((void*)output.data());
    for (auto i = 0; i < std::min(60UL, ns); i++) {
      std::cout << (int32_t)int32_output[i] << " ";
      if ((i + 1) % 6 == 0)
        std::cout << "\n";
    }
    for (auto i = (size_t)(std::max(0, ((int)(ns) - 60))); i < ns; i++) {
      std::cout << (int32_t)int32_output[i] << " ";
      if ((i + 1) % 6 == 0)
        std::cout << "\n";
    }
    std::cout << "\n";
  } else {
    std::cout << "output size = " << output.size() << "\n";
    for (auto i = 0; i < std::min(6UL * 10, output.size()); i++) {
      std::cout << (int32_t)output[i] << " ";
    }
    for (auto i = (size_t)(std::max(0, ((int)(output.size()) - 6 * 10))); i < output.size(); i++) {
      std::cout << (int32_t)output[i] << " ";
    }
    std::cout << "\n";
  }
  return 0;
}
void test_thread() {
  vector<std::thread> threads;
  for (int t = 0; t < 1; ++t) {
    cerr << "threads t = " << t << "\n";
    threads.push_back(thread([&]() {
          string model_root = "/home/lizhen/model_storage/resnet50_v1/data/";
          // model_root = "/home/kaihuo/cortex_fullnode_storage/cifar_resnet20_v2/data";
          // model_root = "/home/lizhen/storage/mnist/data/";
          // model_root = "/home/lizhen/storage/animal10/data";
          // model_root = "/home/kaihuo/cortex_fullnode_storage/imagenet_inceptionV3/data";
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
     "/home/lizhen/model_storage/dcnet_mnist_v1/data",
     "/home/lizhen/model_storage/mobilenetv1.0_imagenet/data",
     "/home/lizhen/model_storage/resnet50_v1_imagenet/data",
     "/home/lizhen/model_storage/animal10/data",
     //"/home/lizhen/model_storage/dcnet_v0_mnist/data",
     "/home/lizhen/model_storage/resnet50_v2/data",
     "/home/lizhen/model_storage/vgg16_gcv/data",
    // "/home/lizhen/model_storage/sentiment_trec/data",
     "/home/lizhen/model_storage/vgg19_gcv/data",
     "/home/lizhen/model_storage/squeezenet_gcv1.1/data",
     "/home/lizhen/model_storage/squeezenet_gcv1.0/data",
     "/home/lizhen/model_storage/octconv_resnet26_0.250/data",
     "/home/lizhen/model_storage/yolo3_darknet53_b1/data"
  };
  for (auto model_root : model_roots) {
    run_LIF(model_root);
  }
}
int main() {
  test_models();
  return 0;
}
