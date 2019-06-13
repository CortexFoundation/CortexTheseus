#include <cvm/c_api.h>
#include <cvm/model.h>
#include <iostream>
#include <thread>
#include <omp.h>
#include <cvm/runtime/registry.h>
#include <cvm/op.h>
#include <cvm/op_attr_types.h>
#include <cvm/runtime/ndarray.h>
#include <cvm/runtime/packed_func.h>
#include <cvm/runtime/registry.h>
#include <cvm/runtime/serializer.h>
#include <cvm/node.h>
#include <cvm/runtime/c_runtime_api.h>
#include "npy.hpp"
#include <string.h>
#include <fstream>
#include <dirent.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;

using cvm::runtime::PackedFunc;
using cvm::runtime::Registry;
using namespace cvm;
using namespace cvm::runtime;

int dtype_code{kDLInt};
int dtype_bits{32};
int dtype_lanes{1};

struct CVMOpParam {
  std::string func_name;
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint32_t flatten_data = false;
  std::string attrs;
};

//int ctx = kDLCPU;
int ctx = kDLGPU;

void LoadOp(string op_type, NodeAttrs& attrs) {
  if (op_type == "null") return;
  attrs.name = op_type;
  std::cerr << "op_type " << op_type << "\n";
  attrs.op = cvm::Op::Get(op_type);
  std::cerr << "op_type =====" << op_type << "\n";
}
void LoadOpAttr(std::string json_, NodeAttrs& attrs) {
  std::istringstream is(json_);
  utils::JSONReader reader(&is);
  reader.Read(&attrs.dict);
  if (attrs.op->attr_parser) {
    attrs.op->attr_parser(&attrs);
  }
}

struct OpArgs {
    std::vector<DLTensor> args;
    std::vector<CVMValue> arg_values;
    std::vector<int> arg_tcodes;
    std::vector<int64_t> shape_data;
};

std::function<void()> get_func(
    const CVMOpParam& param, NodeAttrs* attr,
    const std::vector<DLTensor>& args,
    size_t num_inputs)
{

  struct OpArgs {
    std::vector<DLTensor> args;
    std::vector<CVMValue> arg_values;
    std::vector<int> arg_tcodes;
    std::vector<int64_t> shape_data;
  };

  std::shared_ptr<OpArgs> arg_ptr = std::make_shared<OpArgs>();
  // setup address.
  arg_ptr->args = args;
  if (param.flatten_data) {
    arg_ptr->shape_data.resize(arg_ptr->args.size());
  }
  for (size_t i = 0; i < arg_ptr->args.size(); ++i) {
    CVMValue v;
    DLTensor* t = &(arg_ptr->args[i]);
    v.v_handle = t;
    arg_ptr->arg_values.push_back(v);
    arg_ptr->arg_tcodes.push_back(kArrayHandle);
    if (param.flatten_data) {
      arg_ptr->shape_data[i] = std::accumulate(
          t->shape, t->shape + t->ndim, 1, std::multiplies<int64_t>());
      t->ndim = 1;
      t->shape = &(arg_ptr->shape_data[i]);
    }
  }
  CVMValue t_attr;
  t_attr.v_handle = (void*)attr;
  arg_ptr->arg_values.push_back(t_attr);
  arg_ptr->arg_tcodes.push_back(kHandle);


  auto op = param.func_name;
  int device_type = static_cast<int>(ctx);
  std::string module_name = "cvm.runtime.cvm";
  if (device_type == kDLGPU) module_name += "_cuda";
  module_name += ".";
  auto func = cvm::runtime::Registry::Get(module_name + op);
  VERIFY(func != nullptr) << "function undefined " << module_name + op;
  return [arg_ptr, op, func](){
    CVMRetValue rv;
    CVMArgs targs(
      arg_ptr->arg_values.data(),
      arg_ptr->arg_tcodes.data(),
      static_cast<int>(arg_ptr->arg_values.size())
    );
    func->CallPacked(targs, &rv);
  };

  return [](){};
}
namespace cvm {
namespace runtime {
extern void matrix_mul(const int8_t *a, const int8_t *b, const int32_t *bias,
        int32_t *c, const int M, const int K, const int N, int algo);
}
}
void test_matrix_mul() {
    int M = 7, K = 2, N = 3;
    vector<int8_t> a(M * K), b(K * N);
    vector<int32_t> c(M * N);
    vector<int32_t> bias(M);
    // std::generate(v.begin(), v.end(), [n = 0] () mutable { return n++; });
    std::generate(a.begin(), a.end(), [n = 0] () mutable { return n++; });
    std::generate(b.begin(), b.end(), [n = 0] () mutable { return n++; });
    std::generate(bias.begin(), bias.end(), [n = 0] () mutable { return n++; });
    matrix_mul(a.data(), b.data(), bias.data(), c.data(), M, K, N, 0);
    for (auto x : c) {
        std::cout << x << " ";
    }
    std::cout << "\n";
}
void test_depthwise_conv () {
    string attr_str = " {\"layout\": \"NCHW\", \"kernel_layout\": \"OIHW\", \"kernel_size\": \"[3, 3]\", \"padding\": \"(1, 1)\", \"use_bias\": \"True\", \"strides\": \"(1, 1)\", \"channels\": \"10\", \"dilation\": \"(1, 1)\", \"groups\": \"1024\"} ";
    std::vector<int> dims_ = {4, 4, 4};
    vector<std::vector<int64_t>> shapes_ = {{1, 1024, 7, 7}, {1024, 1, 3, 3}, {1, 1024, 7, 7}};
    CVMOpParam params;
    params.num_inputs = 2;
    params.num_outputs= 1;
    params.func_name = "conv2d";
    std::vector<DLTensor> args(params.num_inputs + params.num_outputs);
    for (uint32_t i = 0; i < args.size(); i++) {
      DLTensor* dl;
      CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, ctx, 1, &dl);
      args[i] = *dl;
    }

    std::vector<unsigned long> tshape;
    std::vector<int32_t> tdata;
    npy::LoadArrayFromNumpy("/tmp/conv2d_depthwise/in.x.npy", tshape, tdata);
    int32_t *dldata = static_cast<int32_t*>(args[0].data);
    memcpy(dldata, tdata.data(), sizeof(int32_t) * tdata.size());
    int n_c = shapes_[0][1];
    int i_h = shapes_[0][2];
    int i_w = shapes_[0][3];
    if (false) {
      for (int c = 0; c < shapes_[0][1]; c++) {
        for (int i = 0; i < shapes_[0][2]; i++) {
          for (int j = 0; j < shapes_[0][3]; j++) {
            std::cerr << dldata[(c) * i_h * i_w +  i * i_w + j] << " ";
          }
          std::cerr << "\n";
        }
        std::cerr << "\n";
      }
    }

    std::vector<unsigned long> tshape2;
    std::vector<int32_t> tdata2;
    int32_t *dldata2 = static_cast<int32_t*>(args[1].data);
    npy::LoadArrayFromNumpy("/tmp/conv2d_depthwise/in.w.npy", tshape2, tdata2);
    memcpy(dldata2, tdata2.data(), sizeof(int32_t) * tdata2.size());

    NodeAttrs attr;
    LoadOp(params.func_name, attr);
    LoadOpAttr(attr_str, attr);
    auto op_slice = get_func(params, &attr, args, params.num_inputs);
    op_slice();

    int32_t *dldata3 = static_cast<int32_t*>(args[2].data);
    std::vector<unsigned long> tshape3;
    std::vector<int32_t> tdata3;
    npy::LoadArrayFromNumpy("/tmp/conv2d_depthwise/out.y.npy", tshape3, tdata3);
    int ret =  memcmp(dldata3, tdata3.data(), sizeof(int32_t) * tdata3.size());
    printf("match %d | %d\n", ret == 0, ret);
    int o_h = shapes_[2][2];
    int o_w = shapes_[2][3];
    if (false) {
    std::cerr << "Expected\n";
    for (int c = 0; c < n_c; c++) {
      for (int i = 0; i < o_h; i++) {
        for (int j = 0; j < o_w; j++) {
          std::cerr << tdata3.data()[(c) * o_h * o_w +  i * o_w + j] << " ";
        }
        std::cerr << "\n";
      }
        std::cerr << "\n";
    }
    std::cerr << "Got\n";
    for (int c = 0; c < n_c; c++) {
      for (int i = 0; i < o_h; i++) {
        for (int j = 0; j < o_w; j++) {
          std::cerr << dldata3[(c ) * o_h * o_w +  i * o_w + j] << " ";
        }
        std::cerr << "\n";
      }
        std::cerr << "\n";
    }
    }
}
void test_transpose() {
    string attr_str = " {\"axes\": \"(1, 2, 0)\"} ";
    std::vector<int> dims_ = {3,  3};
    vector<std::vector<int64_t>> shapes_ = {{38, 32, 300},{32, 300, 38}};
    CVMOpParam params;
    params.num_inputs = 1;
    params.num_outputs= 1;
    params.func_name = "transpose";
    std::vector<DLTensor> args(params.num_inputs + params.num_outputs);
    for (uint32_t i = 0; i < args.size(); i++) {
      DLTensor* dl;
      CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, ctx, 1, &dl);
      args[i] = *dl;
    }

    std::vector<unsigned long> tshape;
    std::vector<int32_t> tdata;
    npy::LoadArrayFromNumpy("/tmp/transpose/in.x.npy", tshape, tdata);
    int32_t *dldata = static_cast<int32_t*>(args[0].data);
    memcpy(dldata, tdata.data(), sizeof(int32_t) * tdata.size());

    NodeAttrs attr;
    LoadOp(params.func_name, attr);
    LoadOpAttr(attr_str, attr);
    auto op_slice = get_func(params, &attr, args, params.num_inputs);
    op_slice();

    int32_t *dldata3 = static_cast<int32_t*>(args[1].data);
    std::vector<unsigned long> tshape3;
    std::vector<int32_t> tdata3;
    npy::LoadArrayFromNumpy("/tmp/transpose/out.y.npy", tshape3, tdata3);
    int ret =  memcmp(dldata3, tdata3.data(), sizeof(int32_t) * tdata3.size());
    printf("match %d | %d\n", ret == 0, ret);

    if (true) {
    std::cerr << "Expected\n";
    for (int c = 0; c < shapes_[0][0] *  shapes_[0][1] *  shapes_[0][2] ; c++) {
        std::cerr << tdata3.data()[c] << " ";
    }
    std::cerr << "\n";
    std::cerr << "Got\n";
    for (int c = 0; c < shapes_[1][0] *  shapes_[1][1] *  shapes_[1][2] ; c++) {
        std::cerr << dldata3[c] << " ";
    }
    std::cerr << "\n";
    }
}
void test_take() {
    string attr_str = " {\"axis\": \"0\"} ";
    CVMOpParam params;
    params.func_name = "take";
    params.num_inputs = 2;
    params.num_outputs= 1;
    std::vector<DLTensor> args(params.num_inputs + params.num_outputs);
    std::vector<std::vector<unsigned long>> tshape(args.size());
    std::vector<std::vector<int32_t>> tdata(args.size());
    npy::LoadArrayFromNumpy("/tmp/take/in.x.npy", tshape[0], tdata[0]);
    npy::LoadArrayFromNumpy("/tmp/take/in.w.npy", tshape[1], tdata[1]);
    npy::LoadArrayFromNumpy("/tmp/take/out.y.npy", tshape[2], tdata[2]);
    vector<std::vector<int64_t>> shapes_(args.size());
    std::vector<int> dims_(args.size());
    for (auto idx = 0; idx < args.size(); idx++) {
      shapes_[idx].resize(tshape[idx].size());
      dims_[idx] = (tshape[idx].size());
      std::cout << tshape[idx].size() << "\n";
      for (auto j = 0; j < shapes_[idx].size(); j++) {
        shapes_[idx][j] = tshape[idx][j];
        std::cout << tshape[idx][j] << " ";
      }
      std::cout << "\n";
    }
    DLTensor* cpu_tensor;
    for (uint32_t i = 0; i < args.size(); i++) {
      DLTensor* dl;
      CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, ctx, 1, &dl);
      args[i] = *dl;
      if (i < params.num_inputs) {
        CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
        memcpy(cpu_tensor->data, tdata[i].data(), sizeof(int32_t) * tdata[i].size());
        CVMArrayCopyFromTo(cpu_tensor, dl, nullptr);
        CVMArrayFree(cpu_tensor);
      }
    }

    NodeAttrs attr;
    LoadOp(params.func_name, attr);
    LoadOpAttr(attr_str, attr);
    auto op_slice = get_func(params, &attr, args, params.num_inputs);
    op_slice();

    vector<int32_t> cpu_output_tensor(tdata[params.num_inputs].size());
    {
      int i = params.num_inputs; // first output
      CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
      CVMArrayCopyFromTo(&args[i], cpu_tensor, nullptr);
      memcpy(cpu_output_tensor.data(), cpu_tensor->data, sizeof(int32_t) * tdata[i].size());
      CVMArrayFree(cpu_tensor);
    }
    int ret =  memcmp(cpu_output_tensor.data(),
                      tdata[params.num_inputs].data(),
                      sizeof(int32_t) * tdata[params.num_inputs].size());
    printf("match %d | %d\n", ret == 0, ret);

    //if (true) {
    //std::cerr << "Expected " << shapes_[0][0] << " " <<  shapes_[0][1] << " " << shapes_[0][2] << "\n";
    //for (int c = 0; c < tdata[params.num_inputs].size() ; c++) {
    //    std::cerr << tdata[params.num_inputs].data()[c] << " ";
    //}
    //std::cerr << "\n";
    //std::cerr << "Got " << shapes_[0][0] << " " <<  shapes_[0][1] << " " << shapes_[0][2] << "\n";
    //for (int c = 0; c < tdata[params.num_inputs].size() ; c++) {
    //    std::cerr << cpu_output_tensor[c] << " ";
    //}
    //std::cerr << "\n";
    //}
}
int findAllSubDir(std::vector<string> &filelist, const char *basePath)
{
    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if ((dir=opendir(basePath)) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        else if(ptr->d_type == 8)    //file
        {
            // //printf("d_name:%s/%s\n",basePath,ptr->d_name);
            // string temp = ptr->d_name;
            // //cout  << temp << endl;
            // string sub = temp.substr(temp.length() - 4, temp.length()-1);
            // //cout  << sub << endl;
            // if(sub == format)
            // {
            //     string path = basePath;
            //     path += "/";
            //     path += ptr->d_name;
            //     filelist.push_back(path);
            // }
        }
        else if(ptr->d_type == 10)    ///link file
        {
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);
        }
        else if(ptr->d_type == 4)    ///dir
        {
            memset(base,'\0',sizeof(base));
            strcpy(base,basePath);
            strcat(base,"/");
            strcat(base,ptr->d_name);
            filelist.push_back(ptr->d_name);
            findAllSubDir(filelist, base);
        }
    }
    closedir(dir);
    return 1;
}
void read_one_line(string filename, string& str){
    ifstream infile;
    infile.open(filename);
    if(!infile.is_open()){
        str = "";
        return;
    }
    getline(infile, str);
    infile.close();
}
template<typename T>
void print(vector<T> &data){
  for(int i = 0; i < data.size(); i++){
    printf("%d ", data[i]);
  }
  printf("\n");

}
const string CASE_DIR = "/data/ops_generator";

void test_op(string op_name, int num_inputs, int num_outputs) {
  printf("\ntest %s\n", op_name.c_str());
	std::vector<string> case_list;
	string case_dir = CASE_DIR + "/" + op_name + "/";
	findAllSubDir(case_list, case_dir.c_str());

  static auto& finfer_shape =
      Op::GetAttr<cvm::FInferNodeEntryAttr<TShape> >("FInferShape");
  const cvm::Op *op = cvm::Op::Get(op_name);
  auto finfer = finfer_shape.get(op, nullptr);
  if (finfer == nullptr) {
    std::cout << "operator " << op_name
      << "has not registered FInferShape";
    return ;
  }

  for(int ci = 0; ci < case_list.size(); ci++){
		string case_path = case_dir + case_list[ci] + "/";
    string attr_path = case_path + "attr.txt";
    string attr_str = "";
    read_one_line(attr_path, attr_str);
    //string attr_str = " {\"axis\": \"" + std::to_string(i-1) + "\"} ";
    std::cout << attr_str << endl;
    CVMOpParam params;
    params.func_name = op_name;
    params.num_inputs = num_inputs;
    params.num_outputs= num_outputs;
    params.flatten_data = false;
    std::vector<DLTensor> args(params.num_inputs + params.num_outputs);
    std::vector<std::vector<unsigned long>> tshape(args.size());
    std::vector<std::vector<int32_t>> tdata(args.size());
    std::vector<TShape> ishape(num_inputs), oshape(num_outputs);
    for(int in_i = 0; in_i < num_inputs; in_i++){
        string in_path = case_path + "in_" +  std::to_string(in_i) + ".npy";
        cout << in_path << endl;
        npy::LoadArrayFromNumpy(in_path, tshape[in_i], tdata[in_i]);
        TShape shp(tshape[in_i].size());
        for (size_t i = 0; i < shp.ndim(); ++i) {
          shp[i] = tshape[in_i][i];
        }
        ishape[in_i] = shp;
        // ishape.emplace_back(shp);
        std::cout << shp << std::endl;
    }
    DLTensor* cpu_tensor;
    for (uint32_t i = 0; i < num_inputs; i++) {
      DLTensor* dl;
      CVMArrayAlloc((int64_t*)tshape[i].data(), tshape[i].size(), dtype_code, dtype_bits, dtype_lanes, ctx, 1, &dl);
      args[i] = *dl;
      //if (i < params.num_inputs) {
        CVMArrayAlloc((int64_t*)tshape[i].data(), tshape[i].size(), dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
        memcpy(cpu_tensor->data, tdata[i].data(), sizeof(int32_t) * tdata[i].size());
        CVMArrayCopyFromTo(cpu_tensor, dl, nullptr);
        CVMArrayFree(cpu_tensor);
      //}
    }

    NodeAttrs attr;
    LoadOp(params.func_name, attr);
    LoadOpAttr(attr_str, attr);

    bool infer_shape_ret;
    string err_path = case_path + "err.txt", err_str = "";
    read_one_line(err_path, err_str);
    try {
      infer_shape_ret = finfer(attr, &ishape, &oshape);
      if(infer_shape_ret){
        std::cout << "FInferShape ishape=[";
        for (auto& shp : ishape) std::cout << shp << ", ";
        std::cout << "] oshape=[";
        for (auto& shp : oshape) std::cout << shp << ", ";
        std::cout << "]\n";
      }
    } catch (const std::exception& e) {
      std::cerr << "FInferShape error with " << e.what() << std::endl;
      infer_shape_ret = false;
    }
    if(infer_shape_ret == false){
      if(err_str == ""){
        string out_path = case_path + "out_0.npy";
        std::cout << out_path << std::endl;
        npy::LoadArrayFromNumpy(out_path, tshape[num_inputs], tdata[num_inputs]);
        print(tdata[num_inputs]);
        assert(false);
      }else{
        cout << endl;
        continue;
      }
    }

    for(int i = 0; i < num_outputs; i++){
			string out_path = case_path + "out_" + std::to_string(i) + ".npy";
			cout << out_path << endl;
			npy::LoadArrayFromNumpy(out_path, tshape[num_inputs+i], tdata[num_inputs+i]);
      int shape_cmp = memcmp(tshape[num_inputs+i].data(), oshape[i].data(), sizeof(int64_t) * tshape[num_inputs+i].size());
      if(shape_cmp != 0){
        print(tshape[num_inputs+i]);
        //print(oshape[i]);
        std::cout << oshape[i] << endl;
      }
      assert(shape_cmp == 0);
      DLTensor* dl;
      CVMArrayAlloc((int64_t*)tshape[num_inputs+i].data(), tshape[num_inputs+i].size(), dtype_code, dtype_bits, dtype_lanes, ctx, 1, &dl);
      args[num_inputs + i] = *dl;
    }

    auto op = get_func(params, &attr, args, params.num_inputs);
    op();

    vector<int32_t> cpu_output_tensor(tdata[params.num_inputs].size());
    {
      int i = params.num_inputs; // first output
      CVMArrayAlloc((int64_t*)tshape[i].data(), tshape[i].size(), dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
      CVMArrayCopyFromTo(&args[i], cpu_tensor, nullptr);
      memcpy(cpu_output_tensor.data(), cpu_tensor->data, sizeof(int32_t) * tdata[i].size());
      CVMArrayFree(cpu_tensor);
    }
    int ret =  memcmp(cpu_output_tensor.data(),
        tdata[params.num_inputs].data(),
        sizeof(int32_t) * tdata[params.num_inputs].size());
    printf("match %d | %d\n", ret == 0, ret);
    if(ret != 0){
      for(int i = 0; i < num_inputs; i++){
        printf("input%d:", i);
        print(tdata[i]);
      }
      printf("correct out:");
      print(tdata[num_inputs]);
      printf("     my out:");
      print(cpu_output_tensor);
    }
    assert(ret == 0);
    printf("\n");
  }
}
int main() {
//    test_op("take", 2, 1);
    // test_op("concatenate", 2, 1);//pass
    // test_op("repeat", 1, 1); //pass
    // test_op("tile", 1, 1); //pass
    // test_op("strided_slice", 1, 1); //pass
		// test_op("slice_like", 2, 1); // pass
    // test_op("max", 1, 1); // pass
    // test_op("sum", 1,1); // pass
    // test_op("upsampling", 1, 1);
    // test_op("elemwise_add", 2, 1);
     test_op("transpose", 1, 1);// 5th case failed
    return 0;
}
