/*!
 * Copyright (c) 2018 by Contributors
 * \file build_config.h
 * \brief Default detection logic for fopen64 and other symbols.
 *        May be overriden by CMake
 * \author KOLANICH
 */
#ifndef CVMUTIL_BUILD_CONFIG_H_
#define CVMUTIL_BUILD_CONFIG_H_

/* default logic for fopen64 */
#if CVMUTIL_USE_FOPEN64 && \
  (!defined(__GNUC__) || (defined __ANDROID__) || (defined __FreeBSD__) \
  || (defined __APPLE__) || ((defined __MINGW32__) && !(defined __MINGW64__)) \
  || (defined __CYGWIN__) )
  #define CVMUTIL_EMIT_FOPEN64_REDEFINE_WARNING
  #define fopen64 std::fopen
#endif

/* default logic for stack trace */
#if (defined(__GNUC__) && !defined(__MINGW32__)\
     && !defined(__sun) && !defined(__SVR4)\
     && !(defined __MINGW64__) && !(defined __ANDROID__))\
     && !defined(__CYGWIN__) && !defined(__EMSCRIPTEN__)
  #define CVMUTIL_LOG_STACK_TRACE 1
  #define CVMUTIL_LOG_STACK_TRACE_SIZE 10
  #define CVMUTIL_EXECINFO_H <execinfo.h>
#endif

/* default logic for detecting existence of nanosleep() */
#if !(defined _WIN32) || (defined __CYGWIN__)
  #define CVMUTIL_NANOSLEEP_PRESENT
#endif

#endif  // CVMUTIL_BUILD_CONFIG_H_
