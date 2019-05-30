/*!
 *  Copyright (c) 2016 by Contributors
 * \file runtime_base.h
 * \brief Base of all C APIs
 */
#ifndef CVM_RUNTIME_RUNTIME_BASE_H_
#define CVM_RUNTIME_RUNTIME_BASE_H_

#include <cvm/runtime/c_runtime_api.h>
#include <stdexcept>

/*! \brief  macro to guard beginning and end section of all functions */
#define API_BEGIN() try {
/*! \brief every function starts with API_BEGIN();
     and finishes with API_END() or API_END_HANDLE_ERROR */
#define API_END() } \
  catch(std::runtime_error &_except_) { return CVMAPIHandleException(_except_); } \
  catch(std::logic_error &_except_) { return CVMAPIHandleLogicException(_except_); \
  } return 0;  // NOLINT(*)
/*!
 * \brief every function starts with API_BEGIN();
 *   and finishes with API_END() or API_END_HANDLE_ERROR
 *   The finally clause contains procedure to cleanup states when an error happens.
 */
#define API_END_HANDLE_ERROR(Finalize) } catch(std::runtime_error &_except_) { Finalize; return CVMAPIHandleException(_except_); } return 0; // NOLINT(*)

/*!
 * \brief handle exception throwed out
 * \param e the exception
 * \return the return value of API after exception is handled
 */
int CVMAPIHandleException(const std::runtime_error &e);
int CVMAPIHandleLogicException(const std::logic_error &e);

#endif  // CVM_RUNTIME_RUNTIME_BASE_H_
