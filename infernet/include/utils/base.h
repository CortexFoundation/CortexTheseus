/*!
 *  Copyright (c) 2015 by Contributors
 * \file base.h
 * \brief defines configuration macros
 */
#ifndef CVMUTIL_BASE_H_
#define CVMUTIL_BASE_H_

/*! \brief whether use glog for logging */
#ifndef CVMUTIL_USE_GLOG
#define CVMUTIL_USE_GLOG 0
#endif

/*!
 * \brief whether throw utils::Error instead of
 *  directly calling abort when FATAL error occured
 *  NOTE: this may still not be perfect.
 *  do not use FATAL and CHECK in destructors
 */
#ifndef CVMUTIL_LOG_FATAL_THROW
#define CVMUTIL_LOG_FATAL_THROW 1
#endif

/*!
 * \brief whether always log a message before throw
 * This can help identify the error that cannot be catched.
 */
#ifndef CVMUTIL_LOG_BEFORE_THROW
#define CVMUTIL_LOG_BEFORE_THROW 0
#endif

/*!
 * \brief Whether to use customized logger,
 * whose output can be decided by other libraries.
 */
#ifndef CVMUTIL_LOG_CUSTOMIZE
#define CVMUTIL_LOG_CUSTOMIZE 0
#endif

/*! \brief whether compile with hdfs support */
#ifndef CVMUTIL_USE_HDFS
#define CVMUTIL_USE_HDFS 0
#endif

/*! \brief whether compile with s3 support */
#ifndef CVMUTIL_USE_S3
#define CVMUTIL_USE_S3 0
#endif

/*! \brief whether or not use parameter server */
#ifndef CVMUTIL_USE_PS
#define CVMUTIL_USE_PS 0
#endif

/*! \brief whether or not use c++11 support */
#ifndef CVMUTIL_USE_CXX11
#if defined(__GXX_EXPERIMENTAL_CXX0X__) || defined(_MSC_VER)
#define CVMUTIL_USE_CXX11 1
#else
#define CVMUTIL_USE_CXX11 (__cplusplus >= 201103L)
#endif
#endif

/*! \brief strict CXX11 support */
#ifndef CVMUTIL_STRICT_CXX11
#if defined(_MSC_VER)
#define CVMUTIL_STRICT_CXX11 1
#else
#define CVMUTIL_STRICT_CXX11 (__cplusplus >= 201103L)
#endif
#endif

/*! \brief Whether cxx11 thread local is supported */
#ifndef CVMUTIL_CXX11_THREAD_LOCAL
#if defined(_MSC_VER)
#define CVMUTIL_CXX11_THREAD_LOCAL (_MSC_VER >= 1900)
#elif defined(__clang__)
#define CVMUTIL_CXX11_THREAD_LOCAL (__has_feature(cxx_thread_local))
#else
#define CVMUTIL_CXX11_THREAD_LOCAL (__cplusplus >= 201103L)
#endif
#endif


/*! \brief whether RTTI is enabled */
#ifndef CVMUTIL_ENABLE_RTTI
#define CVMUTIL_ENABLE_RTTI 1
#endif

/*! \brief whether use fopen64 */
#ifndef CVMUTIL_USE_FOPEN64
#define CVMUTIL_USE_FOPEN64 1
#endif

/// check if g++ is before 4.6
#if CVMUTIL_USE_CXX11 && defined(__GNUC__) && !defined(__clang_version__)
#if __GNUC__ == 4 && __GNUC_MINOR__ < 6
#pragma message("Will need g++-4.6 or higher to compile all"           \
                "the features in utils-core, "                           \
                "compile without c++0x, some features may be disabled")
#undef CVMUTIL_USE_CXX11
#define CVMUTIL_USE_CXX11 0
#endif
#endif

/*!
 * \brief Use little endian for binary serialization
 *  if this is set to 0, use big endian.
 */
#ifndef CVMUTIL_IO_USE_LITTLE_ENDIAN
#define CVMUTIL_IO_USE_LITTLE_ENDIAN 1
#endif

/*!
 * \brief Enable std::thread related modules,
 *  Used to disable some module in mingw compile.
 */
#ifndef CVMUTIL_ENABLE_STD_THREAD
#define CVMUTIL_ENABLE_STD_THREAD CVMUTIL_USE_CXX11
#endif

/*! \brief whether enable regex support, actually need g++-4.9 or higher*/
#ifndef CVMUTIL_USE_REGEX
#define CVMUTIL_USE_REGEX CVMUTIL_STRICT_CXX11
#endif

/*! \brief helper macro to supress unused warning */
#if defined(__GNUC__)
#define CVMUTIL_ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define CVMUTIL_ATTRIBUTE_UNUSED
#endif

/*! \brief helper macro to generate string concat */
#define CVMUTIL_STR_CONCAT_(__x, __y) __x##__y
#define CVMUTIL_STR_CONCAT(__x, __y) CVMUTIL_STR_CONCAT_(__x, __y)

/*!
 * \brief Disable copy constructor and assignment operator.
 *
 * If C++11 is supported, both copy and move constructors and
 * assignment operators are deleted explicitly. Otherwise, they are
 * only declared but not implemented. Place this macro in private
 * section if C++11 is not available.
 */
#ifndef DISALLOW_COPY_AND_ASSIGN
#  if CVMUTIL_USE_CXX11
#    define DISALLOW_COPY_AND_ASSIGN(T) \
       T(T const&) = delete; \
       T(T&&) = delete; \
       T& operator=(T const&) = delete; \
       T& operator=(T&&) = delete
#  else
#    define DISALLOW_COPY_AND_ASSIGN(T) \
       T(T const&); \
       T& operator=(T const&)
#  endif
#endif

#ifdef __APPLE__
#  define off64_t off_t
#endif

#ifdef _MSC_VER
#if _MSC_VER < 1900
// NOTE: sprintf_s is not equivalent to snprintf,
// they are equivalent when success, which is sufficient for our case
#define snprintf sprintf_s
#define vsnprintf vsprintf_s
#endif
#else
#ifdef _FILE_OFFSET_BITS
#if _FILE_OFFSET_BITS == 32
#pragma message("Warning: FILE OFFSET BITS defined to be 32 bit")
#endif
#endif

extern "C" {
#include <sys/types.h>
}
#endif

#ifdef _MSC_VER
//! \cond Doxygen_Suppress
typedef signed char int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned char uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
//! \endcond
#else
#include <inttypes.h>
#endif
#include <string>
#include <vector>

#if defined(_MSC_VER) && _MSC_VER < 1900
#define noexcept_true throw ()
#define noexcept_false
#define noexcept(a) noexcept_##a
#endif

#if CVMUTIL_USE_CXX11
#define CVMUTIL_THROW_EXCEPTION noexcept(false)
#define CVMUTIL_NO_EXCEPTION  noexcept(true)
#else
#define CVMUTIL_THROW_EXCEPTION
#define CVMUTIL_NO_EXCEPTION
#endif

/*! \brief namespace for utils */
namespace utils {
/*!
 * \brief safely get the beginning address of a vector
 * \param vec input vector
 * \return beginning address of a vector
 */
template<typename T>
inline T *BeginPtr(std::vector<T> &vec) {  // NOLINT(*)
  if (vec.size() == 0) {
    return NULL;
  } else {
    return &vec[0];
  }
}
/*!
 * \brief get the beginning address of a const vector
 * \param vec input vector
 * \return beginning address of a vector
 */
template<typename T>
inline const T *BeginPtr(const std::vector<T> &vec) {
  if (vec.size() == 0) {
    return NULL;
  } else {
    return &vec[0];
  }
}
/*!
 * \brief get the beginning address of a string
 * \param str input string
 * \return beginning address of a string
 */
inline char* BeginPtr(std::string &str) {  // NOLINT(*)
  if (str.length() == 0) return NULL;
  return &str[0];
}
/*!
 * \brief get the beginning address of a const string
 * \param str input string
 * \return beginning address of a string
 */
inline const char* BeginPtr(const std::string &str) {
  if (str.length() == 0) return NULL;
  return &str[0];
}
}  // namespace utils

#if defined(_MSC_VER) && _MSC_VER < 1900
#define constexpr const
#define alignof __alignof
#endif

/* If fopen64 is not defined by current machine,
   replace fopen64 with std::fopen. Also determine ability to print stack trace
   for fatal error and define CVMUTIL_LOG_STACK_TRACE if stack trace can be
   produced. Always keep this #include at the bottom of utils/base.h */
#include <utils/build_config.h>

#endif  // CVMUTIL_BASE_H_
