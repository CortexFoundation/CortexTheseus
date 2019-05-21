/*!
 *  Copyright (c) 2017 by Contributors
 * \file endian.h
 * \brief Endian testing, need c++11
 */
#ifndef CVMUTIL_ENDIAN_H_
#define CVMUTIL_ENDIAN_H_

#include "./base.h"

#ifdef CVMUTIL_CMAKE_LITTLE_ENDIAN
  // If compiled with CMake, use CMake's endian detection logic
  #define CVMUTIL_LITTLE_ENDIAN CVMUTIL_CMAKE_LITTLE_ENDIAN
#else
  #if defined(__APPLE__) || defined(_WIN32)
    #define CVMUTIL_LITTLE_ENDIAN 1
  #elif defined(__GLIBC__)
    #include <endian.h>
    #define CVMUTIL_LITTLE_ENDIAN (__BYTE_ORDER == __LITTLE_ENDIAN)
  #elif defined(__FreeBSD__)
    #include <sys/endian.h>
    #define CVMUTIL_LITTLE_ENDIAN (_BYTE_ORDER == _LITTLE_ENDIAN)
  #elif defined(__EMSCRIPTEN__)
    #define CVMUTIL_LITTLE_ENDIAN 1
  #else
    #error "Unable to determine endianness of your machine; use CMake to compile"
  #endif
#endif

/*! \brief whether serialize using little endian */
#define CVMUTIL_IO_NO_ENDIAN_SWAP (CVMUTIL_LITTLE_ENDIAN == CVMUTIL_IO_USE_LITTLE_ENDIAN)

namespace utils {

/*!
 * \brief A generic inplace byte swapping function.
 * \param data The data pointer.
 * \param elem_bytes The number of bytes of the data elements
 * \param num_elems Number of elements in the data.
 * \note Always try pass in constant elem_bytes to enable
 *       compiler optimization
 */
inline void ByteSwap(void* data, size_t elem_bytes, size_t num_elems) {
  for (size_t i = 0; i < num_elems; ++i) {
    uint8_t* bptr = reinterpret_cast<uint8_t*>(data) + elem_bytes * i;
    for (size_t j = 0; j < elem_bytes / 2; ++j) {
      uint8_t v = bptr[elem_bytes - 1 - j];
      bptr[elem_bytes - 1 - j] = bptr[j];
      bptr[j] = v;
    }
  }
}

}  // namespace utils
#endif  // CVMUTIL_ENDIAN_H_
