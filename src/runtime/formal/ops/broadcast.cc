#include "ops.h"

namespace cvm {
namespace runtime {

double cvm_op_broadcast_cnt = 0;

typedef std::function<int32_t(int32_t a, int32_t b)> broadcast_func;


static void broadcast(cvm::runtime::CVMArgValue& A, 
                      cvm::runtime::CVMArgValue& B, 
                      cvm::runtime::CVMArgValue& Y, 
                      broadcast_func const &f){
    // inputs: A, B
    // outputs: Y
    // A.shape = (m_0, m_1,,, m_{M-1})
    // B.shape = (n_0, n_1,,, n_{N-1})
    auto A_shape = CVMArgShape(A);
    auto B_shape = CVMArgShape(B);
    auto Y_shape = CVMArgShape(Y);

    // K = max(M, N)
    auto K = Y_shape.size();
    
    // SA represents the new shape of A, that is obtained through 
    // extending A.shape into K dimension by prefixing with 1 
    // SA_i = 1, i < K - M
    std::vector<int64_t> SA(K - A_shape.size(), 1);
    // SA_i = m_{i-K+M}, i >= K - M
    SA.insert(SA.end(), A_shape.begin(), A_shape.end());
    
    // SB represents the new shape of B, that is obtained through 
    // extending B.shape into K dimension by prefixing with 1 
    // SB_i = 1, i < K - N
    std::vector<int64_t> SB(K - B_shape.size(), 1);
    // SB_i = n_{i-K+N}, i >= K-N
    SB.insert(SB.end(), B_shape.begin(), B_shape.end());
    
    // d_k, a_k, b_k represent the coordinate index of Y.shape, SA, SB, respectively
    std::vector<int64_t> d_k(K, 0), a_k(K, 0), b_k(K, 0);
    auto a = CVMArg2Data<int32_t>(A); 
    auto b = CVMArg2Data<int32_t>(B); 
    auto c = CVMArg2Data<int32_t>(Y); 
    
    // Y.shape = (k_0, k_1,,, k_{K-1}), k_i = max(SA_i, SB_i)  
    // For \forall i \in [0, K)], d_{i} \in [0, k_{i})
    for (auto j = CVMShapeBegin(Y); j < CVMShapeEnd(Y); j++){
      for (auto i = 0; i < K; i++){
        // a_i = min(d_{i}, SA_i-1)
        a_k[i] = (SA[i] - 1 < d_k[i]) ? SA[i] - 1 : d_k[i];

        // b_i = min(d_{i}, SB_i-1)
        b_k[i] = (SB[i] - 1 < d_k[i]) ? SB[i] - 1 : d_k[i];
      }
      
      // index0 = the number of (a_0, a_1,,, a_{K-1}) on decimal digit
      int index0 = Index2Number(SA, a_k);
      // index1 = the number of (b_0, b_1,,, b_{K-1}) on decimal digit
      int index1 = Index2Number(SB, b_k);
      // index2 = the number of (d_0, d_1,,, d_{K-1}) on decimal digit
      int index2 = Index2Number(Y_shape, d_k);
      
      // Y[d_0, d_1,,, d_{K-1}] = f(A[a_0, a_1,,, a_{K-1}], B[b_0, b_1,,, b_{K-1}])
      c[index2] = f(a[index0], b[index1]);

      IndexBaseShapeAddOne(Y_shape, d_k);
    }

}

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_add")
.set_body([](CVMArgs args, CVMRetValue *ret){
    auto args0 = args[0];
    auto args1 = args[1];
    auto args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return a + b;
    };

    broadcast(args0, args1, args2, f);
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_sub")
.set_body([](CVMArgs args, CVMRetValue *ret){
    auto args0 = args[0];
    auto args1 = args[1];
    auto args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return a - b;
    };

    broadcast(args0, args1, args2, f);
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_mul")
.set_body([](CVMArgs args, CVMRetValue *ret){
    auto args0 = args[0];
    auto args1 = args[1];
    auto args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return a * b;
    };

    broadcast(args0, args1, args2, f);
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_max")
.set_body([](CVMArgs args, CVMRetValue *ret){
    auto args0 = args[0];
    auto args1 = args[1];
    auto args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return a > b ? a : b;
    };

    broadcast(args0, args1, args2, f);
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_div")
.set_body([](CVMArgs args, CVMRetValue *ret){
    auto args0 = args[0];
    auto args1 = args[1];
    auto args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return b == 0 ? 0 : a/b;
    };

    broadcast(args0, args1, args2, f);
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_greater")
.set_body([](CVMArgs args, CVMRetValue *ret){
    auto args0 = args[0];
    auto args1 = args[1];
    auto args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return a > b;
    };

    broadcast(args0, args1, args2, f);
});
}
}

