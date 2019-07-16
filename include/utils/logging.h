/*!
 *  Copyright (c) 2015 by Contributors
 * \file logging.h
 * \brief defines logging macros of utils
 *  allows use of GLOG, fall back to internal
 *  implementation when disabled
 */
#ifndef CVMUTIL_LOGGING_H_
#define CVMUTIL_LOGGING_H_
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>
#include <exception>
#include <memory>
#include "./base.h"

#if CVMUTIL_LOG_STACK_TRACE
#include <cxxabi.h>
#include CVMUTIL_EXECINFO_H
#endif

namespace utils {
/*!
 * \brief exception class that will be thrown by
 *  default logger if CVMUTIL_LOG_FATAL_THROW == 1
 */
struct Error : public std::runtime_error {
  /*!
   * \brief constructor
   * \param s the error message
   */
  explicit Error(const std::string &s) : std::runtime_error(s) {}
};

}  // namespace utils

#if CVMUTIL_USE_GLOG
#else
// use a light version of glog
#include <assert.h>
#include <iostream>
#include <sstream>
#include <ctime>

#if defined(_MSC_VER)
#pragma warning(disable : 4722)
#pragma warning(disable : 4068)
#endif

namespace utils {
inline void InitLogging(const char*) {
  // DO NOTHING
}

class LogCheckError {
 public:
  LogCheckError() : str(nullptr) {}
  explicit LogCheckError(const std::string& str_) : str(new std::string(str_)) {}
  ~LogCheckError() { if (str != nullptr) delete str; }
  operator bool() {return str != nullptr; }
  std::string* str;
};

#ifndef CVMUTIL_GLOG_DEFINED

#define DEFINE_VERIFY_FUNC(name, op)                                    \
  template <typename X, typename Y>                                     \
  inline LogCheckError ValueVerify##name(const X& x, const Y& y) {      \
    if (x op y) return LogCheckError();                                 \
    std::ostringstream os;                                              \
    os << " (" << x << " vs. " << y << ") ";  /* CHECK_XX(x, y) requires x and y can be serialized to string. Use CHECK(x OP y) otherwise. NOLINT(*) */ \
    return LogCheckError(os.str());                                     \
  }                                                                     \
  inline LogCheckError ValueVerify##name(int x, int y) {                \
    return ValueVerify##name<int, int>(x, y);                           \
  }

#define VERIFY_BINARY_OP(name, op, x, y)                                     \
  if (utils::LogCheckError _check_err = utils::ValueVerify##name(x, y))      \
    utils::ValueVerifyFatal(__FILE__, __LINE__).stream()                     \
      << "Check failed: " << #x " " #op " " #y << *(_check_err.str)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
DEFINE_VERIFY_FUNC(_LT, <)
DEFINE_VERIFY_FUNC(_GT, >)
DEFINE_VERIFY_FUNC(_LE, <=)
DEFINE_VERIFY_FUNC(_GE, >=)
DEFINE_VERIFY_FUNC(_EQ, ==)
DEFINE_VERIFY_FUNC(_NE, !=)
#pragma GCC diagnostic pop

// Always-on checking
#define VERIFY(x)                                            \
  if (!(x))                                                  \
    utils::ValueVerifyFatal(__FILE__, __LINE__).stream()     \
      << "Check failed: " #x << ' '

#define VERIFY_LT(x, y) VERIFY_BINARY_OP(_LT, <, x, y)
#define VERIFY_GT(x, y) VERIFY_BINARY_OP(_GT, >, x, y)
#define VERIFY_LE(x, y) VERIFY_BINARY_OP(_LE, <=, x, y)
#define VERIFY_GE(x, y) VERIFY_BINARY_OP(_GE, >=, x, y)
#define VERIFY_EQ(x, y) VERIFY_BINARY_OP(_EQ, ==, x, y)
#define VERIFY_NE(x, y) VERIFY_BINARY_OP(_NE, !=, x, y)
#define VERIFY_NOTNULL(x) \
  ((x) == NULL ? utils::ValueVerifyFatal(__FILE__, __LINE__).stream() << "Check  notnull: "  #x << ' ', (x) : (x)) // NOLINT(*)

#define DEFINE_CHECK_FUNC(name, op)                               \
  template <typename X, typename Y>                               \
  inline LogCheckError LogCheck##name(const X& x, const Y& y) {   \
    if (x op y) return LogCheckError();                           \
    std::ostringstream os;                                        \
    os << " (" << x << " vs. " << y << ") ";  /* CHECK_XX(x, y) requires x and y can be serialized to string. Use CHECK(x OP y) otherwise. NOLINT(*) */ \
    return LogCheckError(os.str());                               \
  }                                                               \
  inline LogCheckError LogCheck##name(int x, int y) {             \
    return LogCheck##name<int, int>(x, y);                        \
  }

#define CHECK_BINARY_OP(name, op, x, y)                                \
  if (utils::LogCheckError _check_err = utils::LogCheck##name(x, y))   \
    utils::LogMessageFatal(__FILE__, __LINE__).stream()                \
      << "Check failed: " << #x " " #op " " #y << *(_check_err.str)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
DEFINE_CHECK_FUNC(_LT, <)
DEFINE_CHECK_FUNC(_GT, >)
DEFINE_CHECK_FUNC(_LE, <=)
DEFINE_CHECK_FUNC(_GE, >=)
DEFINE_CHECK_FUNC(_EQ, ==)
DEFINE_CHECK_FUNC(_NE, !=)
#pragma GCC diagnostic pop

// Always-on checking
#define CHECK(x)                                            \
  if (!(x))                                                 \
    utils::LogMessageFatal(__FILE__, __LINE__).stream()     \
      << "Check failed: " #x << ' '

#define CHECK_LT(x, y) CHECK_BINARY_OP(_LT, <, x, y)
#define CHECK_GT(x, y) CHECK_BINARY_OP(_GT, >, x, y)
#define CHECK_LE(x, y) CHECK_BINARY_OP(_LE, <=, x, y)
#define CHECK_GE(x, y) CHECK_BINARY_OP(_GE, >=, x, y)
#define CHECK_EQ(x, y) CHECK_BINARY_OP(_EQ, ==, x, y)
#define CHECK_NE(x, y) CHECK_BINARY_OP(_NE, !=, x, y)
#define CHECK_NOTNULL(x) \
  ((x) == NULL ? utils::LogMessageFatal(__FILE__, __LINE__).stream() << "Check  notnull: "  #x << ' ', (x) : (x)) // NOLINT(*)
// Debug-only checking.
#define DCHECK(x) \
  while (false) CHECK(x)
#define DCHECK_LT(x, y) \
  while (false) CHECK((x) < (y))
#define DCHECK_GT(x, y) \
  while (false) CHECK((x) > (y))
#define DCHECK_LE(x, y) \
  while (false) CHECK((x) <= (y))
#define DCHECK_GE(x, y) \
  while (false) CHECK((x) >= (y))
#define DCHECK_EQ(x, y) \
  while (false) CHECK((x) == (y))
#define DCHECK_NE(x, y) \
  while (false) CHECK((x) != (y))

#define LOG_INFO utils::LogMessage()
#define LOG_WARNING LOG_INFO
#define LOG_ERROR LOG_INFO
#define LOG_FATAL utils::LogMessageFatal(__FILE__, __LINE__)
#define LOG_QFATAL LOG_FATAL

// Poor man version of VLOG
#define VLOG(x) LOG_INFO.stream()

#define LOG(severity) LOG_##severity.stream()
#define LG LOG_INFO.stream()
#define LOG_IF(severity, condition) \
  !(condition) ? (void)0 : utils::LogMessageVoidify() & LOG(severity)

#define LOG_DFATAL LOG_FATAL
#define DFATAL FATAL
#define DLOG(severity) LOG(severity)
#define DLOG_IF(severity, condition) LOG_IF(severity, condition)

// Poor man version of LOG_EVERY_N
#define LOG_EVERY_N(severity, n) LOG(severity)

#endif  // CVMUTIL_GLOG_DEFINED

class DateLogger {
 public:
  DateLogger() {
#if defined(_MSC_VER)
    _tzset();
#endif
  }
  const char* HumanDate() {
#ifndef _LIBCPP_SGX_CONFIG
#if defined(_MSC_VER)
    _strtime_s(buffer_, sizeof(buffer_));
#else
    time_t time_value = time(NULL);
    struct tm *pnow;
#if !defined(_WIN32)
    struct tm now;
    pnow = localtime_r(&time_value, &now);
#else
    pnow = localtime(&time_value);  // NOLINT(*)
#endif
    snprintf(buffer_, sizeof(buffer_), "%02d:%02d:%02d",
             pnow->tm_hour, pnow->tm_min, pnow->tm_sec);
#endif
#endif  // _LIBCPP_SGX_CONFIG
    return buffer_;
  }

 private:
  char buffer_[9];
};

class LogMessage {
 public:
  LogMessage()
      :
#ifdef __ANDROID__
        log_stream_(std::cout)
#else
        log_stream_(std::cerr)
#endif
  { }
  ~LogMessage() {  log_stream_ << '\n'; }
  std::ostream& stream() { return log_stream_; }

 protected:
  std::ostream& log_stream_;

 private:
  LogMessage(const LogMessage&);
  void operator=(const LogMessage&);
};

#if CVMUTIL_LOG_STACK_TRACE
inline std::string Demangle(void *trace, char const *msg_str) {
  using std::string;
  string msg(msg_str);
  size_t symbol_start = string::npos;
  size_t symbol_end = string::npos;
  if ( ((symbol_start = msg.find("_Z")) != string::npos)
       && (symbol_end = msg.find_first_of(" +", symbol_start)) ) {
    string left_of_symbol(msg, 0, symbol_start);
    string symbol(msg, symbol_start, symbol_end - symbol_start);
    string right_of_symbol(msg, symbol_end);

    int status = 0;
    size_t length = string::npos;
    std::unique_ptr<char, decltype(&std::free)> demangled_symbol =
            {abi::__cxa_demangle(symbol.c_str(), 0, &length, &status), &std::free};
    if (demangled_symbol && status == 0 && length > 0) {
      string symbol_str(demangled_symbol.get());
      std::ostringstream os;
      os << left_of_symbol << symbol_str << right_of_symbol;
      return os.str();
    }
  }
  return string(msg_str);
}

inline std::string StackTrace(
    const size_t stack_size = CVMUTIL_LOG_STACK_TRACE_SIZE) {
  using std::string;
  std::ostringstream stacktrace_os;
  std::vector<void*> stack(stack_size);
  int nframes = backtrace(stack.data(), static_cast<int>(stack_size));
  stacktrace_os << "Stack trace returned " << nframes << " entries:" << std::endl;
  char **msgs = backtrace_symbols(stack.data(), nframes);
  if (msgs != nullptr) {
    for (int frameno = 0; frameno < nframes; ++frameno) {
      string msg = utils::Demangle(stack[frameno], msgs[frameno]);
      stacktrace_os << "[bt] (" << frameno << ") " << msg << "\n";
    }
  }
  free(msgs);
  string stack_trace = stacktrace_os.str();
  return stack_trace;
}

#else  // CVMUTIL_LOG_STACK_TRACE is off

inline std::string demangle(char const* msg_str) {
  return std::string();
}

inline std::string StackTrace(const size_t stack_size = 0) {
  return std::string("stack traces not available when "
  "CVMUTIL_LOG_STACK_TRACE is disabled at compile time.");
}

#endif  // CVMUTIL_LOG_STACK_TRACE

class ValueVerifyFatal {
 public:
  ValueVerifyFatal(const char* file, int line) {
    log_stream_ << "[" << pretty_date_.HumanDate() << "] " << file << ":"
                << line << ": ";
    log_stream_ << "\n\n";
 }
  std::ostringstream &stream() { return log_stream_; }
  ~ValueVerifyFatal() noexcept(false) {
    // throwing out of destructor is evil
    // hopefully we can do it here
    // also log the message before throw
    LOG(ERROR) << log_stream_.str();
    throw std::logic_error(log_stream_.str());
  }

 private:
  std::ostringstream log_stream_;
  DateLogger pretty_date_;
  ValueVerifyFatal(const ValueVerifyFatal&);
  void operator=(const ValueVerifyFatal&);
};

class LogMessageFatal {
 public:
  LogMessageFatal(const char* file, int line) {
    log_stream_ << "[" << pretty_date_.HumanDate() << "] " << file << ":"
                << line << ": ";
    log_stream_ << "\n\n" << StackTrace() << "\n";

    // throwing out of destructor is evil
    // hopefully we can do it here
    // also log the message before throw
 }
  std::ostringstream &stream() { return log_stream_; }
  ~LogMessageFatal() noexcept(false) {
    LOG(ERROR) << log_stream_.str();
    throw std::runtime_error(log_stream_.str());
  }

 private:
  std::ostringstream log_stream_;
  DateLogger pretty_date_;
  LogMessageFatal(const LogMessageFatal&);
  void operator=(const LogMessageFatal&);
};

// This class is used to explicitly ignore values in the conditional
// logging macros.  This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class LogMessageVoidify {
 public:
  LogMessageVoidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than "?:". See its usage.
  void operator&(std::ostream&) {}
};

}  // namespace utils

#endif
#endif  // CVMUTIL_LOGGING_H_
