/*!
 *  Copyright (c) 2016 by Contributors
 * \file runtime_base.h
 * \brief Base of all C APIs
 */
#ifndef CVM_ERRORS_H
#define CVM_ERRORS_H

#include <utils/thread_local.h>
#include <cvm/c_api.h>
#include <cvm/runtime/c_runtime_api.h>
#include <cvm/symbolic.h>
#include <exception>

#define PRINT(e) printf("ERROR: %s\n", e);
#define API_BEGIN() try {

#define API_END_HANDLE_ERROR(finalize) \
  } catch (const std::runtime_error &e) { \
    PRINT(e.what()); finalize; return ERROR_RUNTIME; \
  } catch (const std::logic_error &e) { \
    PRINT(e.what()); finalize; return ERROR_LOGIC; \
  } catch (const std::exception &e) { \
    PRINT(e.what()); finalize; return ERROR_RUNTIME; \
  } \
  return SUCCEED;

#define API_END() \
  } catch (const std::runtime_error &e) { \
    PRINT(e.what()); return ERROR_RUNTIME; \
  } catch (const std::logic_error &e) { \
    PRINT(e.what()); return ERROR_LOGIC; \
  } catch (const std::exception &e) { \
    PRINT(e.what()); return ERROR_RUNTIME; \
  } \
  return SUCCEED;

struct CVMRuntimeEntry {
  std::string ret_str;
  std::string last_error;
  CVMByteArray ret_bytes;
};

typedef utils::ThreadLocalStore<CVMRuntimeEntry> CVMAPIRuntimeStore;

struct CVMAPIThreadLocalEntry {
  /*! \brief result holder for returning string */
  std::string ret_str;
  /*! \brief result holder for returning strings */
  std::vector<std::string> ret_vec_str;
  /*! \brief result holder for returning string pointers */
  std::vector<const char *> ret_vec_charp;
  /*! \brief result holder for returning handles */
  std::vector<void *> ret_handles;
  /*! \brief argument holder to hold symbol */
  std::unordered_map<std::string, const cvm::Symbol*> kwarg_symbol;
};

/*! \brief Thread local store that can be used to hold return values. */
typedef utils::ThreadLocalStore<CVMAPIThreadLocalEntry> CVMAPIThreadLocalStore;

/*!
 * \brief Used for implementing C API function.
 *  Set last error message before return.
 * \param msg The error message to be set.
 */
CVM_DLL void CVMAPISetLastError(const char* msg);

/*!
 * \brief return str message of the last error
 *  all function in this file will return 0 when success
 *  and -1 when an error occured,
 *  CVMGetLastError can be called to retrieve the error
 *
 *  this function is threadsafe and can be called by different thread
 *  \return error info
 */
CVM_DLL const char *CVMGetLastError(void);

int CVMAPIHandleException(const std::runtime_error &e);
int CVMAPIHandleLogicException(const std::logic_error &e);

#endif  // CVM_ERRORS_H
