/*!
 *  Copyright (c) 2016 by Contributors
 * \file cvm/base.h
 * \brief Configuration of cvm as well as basic data structure.
 */
#ifndef CVM_BASE_H_
#define CVM_BASE_H_

#include <utils/base.h>
#include <utils/common.h>
#include <utils/any.h>
#include <utils/memory.h>
#include <utils/logging.h>
#include <utils/registry.h>
#include <utils/array_view.h>

namespace cvm {

/*! \brief any type */
using utils::any;

/*! \brief array_veiw type  */
using utils::array_view;

/*!\brief getter function of any type */
using utils::get;

/*!\brief "unsafe" getter function of any type */
using utils::unsafe_get;

}  // namespace cvm

// describe op registration point
#define CVM_STRINGIZE_DETAIL(x) #x
#define CVM_STRINGIZE(x) CVM_STRINGIZE_DETAIL(x)
#define CVM_DESCRIBE(...) describe(__VA_ARGS__ "\n\nFrom:" __FILE__ ":" CVM_STRINGIZE(__LINE__))
#define CVM_ADD_FILELINE "\n\nDefined in " __FILE__ ":L" CVM_STRINGIZE(__LINE__)
#endif  // CVM_BASE_H_
