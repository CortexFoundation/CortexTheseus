#ifndef CVM_TIME_H
#define CVM_TIME_H

#include <chrono>
#include <iomanip>

using cvm_clock = std::chrono::high_resolution_clock;

using std::chrono::nanoseconds;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::minutes;
using std::chrono::hours;

#ifdef PROFILE

#define TIME_INIT(id) \
  auto __start_ ## id = cvm_clock::now();

#else
#define TIME_INIT(id)
#endif

#ifdef PROFILE

#define TIME_ELAPSED(id, msg) \
  auto __end_ ## id = cvm_clock::now(); \
  auto __count_ ## id = __end_ ## id - \
    __start_ ## id; \
  std::cout << "Time elapsed: " \
    << std::setw(10) << std::setprecision(3) \
    << (double)(__count_ ## id.count()) / 1000000 \
    << " ms in " << msg << std::endl;

#else
#define TIME_ELAPSED(id, msg)
#endif

#endif // CVM_TIME_H
