/*!
 *  Copyright (c) 2015 by Contributors
 * \file type_traits.h
 * \brief type traits information header
 */
#ifndef CVMUTIL_TYPE_TRAITS_H_
#define CVMUTIL_TYPE_TRAITS_H_

#include "./base.h"
#if CVMUTIL_USE_CXX11
#include <type_traits>
#endif
#include <string>

namespace utils {
/*!
 * \brief whether a type is pod type
 * \tparam T the type to query
 */
template<typename T>
struct is_pod {
#if CVMUTIL_USE_CXX11
  /*! \brief the value of the traits */
  static const bool value = std::is_pod<T>::value;
#else
  /*! \brief the value of the traits */
  static const bool value = false;
#endif
};


/*!
 * \brief whether a type is integer type
 * \tparam T the type to query
 */
template<typename T>
struct is_integral {
#if CVMUTIL_USE_CXX11
  /*! \brief the value of the traits */
  static const bool value = std::is_integral<T>::value;
#else
  /*! \brief the value of the traits */
  static const bool value = false;
#endif
};

/*!
 * \brief whether a type is floating point type
 * \tparam T the type to query
 */
template<typename T>
struct is_floating_point {
#if CVMUTIL_USE_CXX11
  /*! \brief the value of the traits */
  static const bool value = std::is_floating_point<T>::value;
#else
  /*! \brief the value of the traits */
  static const bool value = false;
#endif
};

/*!
 * \brief whether a type is arithemetic type
 * \tparam T the type to query
 */
template<typename T>
struct is_arithmetic {
#if CVMUTIL_USE_CXX11
  /*! \brief the value of the traits */
  static const bool value = std::is_arithmetic<T>::value;
#else
  /*! \brief the value of the traits */
  static const bool value = (utils::is_integral<T>::value ||
                             utils::is_floating_point<T>::value);
#endif
};

/*!
 * \brief helper class to construct a string that represents type name
 *
 * Specialized this class to defined type name of custom types
 *
 * \tparam T the type to query
 */
template<typename T>
struct type_name_helper {
  /*!
   * \return a string of typename.
   */
  static inline std::string value() {
    return "";
  }
};

/*!
 * \brief the string representation of type name
 * \tparam T the type to query
 * \return a const string of typename.
 */
template<typename T>
inline std::string type_name() {
  return type_name_helper<T>::value();
}

/*!
 * \brief whether a type have save/load function
 * \tparam T the type to query
 */
template<typename T>
struct has_saveload {
  /*! \brief the value of the traits */
  static const bool value = false;
};

/*!
 * \brief template to select type based on condition
 * For example, IfThenElseType<true, int, float>::Type will give int
 * \tparam cond the condition
 * \tparam Then the typename to be returned if cond is true
 * \tparam Else typename to be returned if cond is false
*/
template<bool cond, typename Then, typename Else>
struct IfThenElseType;

/*! \brief macro to quickly declare traits information */
#define CVMUTIL_DECLARE_TRAITS(Trait, Type, Value)       \
  template<>                                          \
  struct Trait<Type> {                                \
    static const bool value = Value;                  \
  }

/*! \brief macro to quickly declare traits information */
#define CVMUTIL_DECLARE_TYPE_NAME(Type, Name)            \
  template<>                                          \
  struct type_name_helper<Type> {                     \
    static inline std::string value() {               \
      return Name;                                    \
    }                                                 \
  }

//! \cond Doxygen_Suppress
// declare special traits when C++11 is not available
#if CVMUTIL_USE_CXX11 == 0
CVMUTIL_DECLARE_TRAITS(is_pod, char, true);
CVMUTIL_DECLARE_TRAITS(is_pod, int8_t, true);
CVMUTIL_DECLARE_TRAITS(is_pod, int16_t, true);
CVMUTIL_DECLARE_TRAITS(is_pod, int32_t, true);
CVMUTIL_DECLARE_TRAITS(is_pod, int64_t, true);
CVMUTIL_DECLARE_TRAITS(is_pod, uint8_t, true);
CVMUTIL_DECLARE_TRAITS(is_pod, uint16_t, true);
CVMUTIL_DECLARE_TRAITS(is_pod, uint32_t, true);
CVMUTIL_DECLARE_TRAITS(is_pod, uint64_t, true);
CVMUTIL_DECLARE_TRAITS(is_pod, float, true);
CVMUTIL_DECLARE_TRAITS(is_pod, double, true);

CVMUTIL_DECLARE_TRAITS(is_integral, char, true);
CVMUTIL_DECLARE_TRAITS(is_integral, int8_t, true);
CVMUTIL_DECLARE_TRAITS(is_integral, int16_t, true);
CVMUTIL_DECLARE_TRAITS(is_integral, int32_t, true);
CVMUTIL_DECLARE_TRAITS(is_integral, int64_t, true);
CVMUTIL_DECLARE_TRAITS(is_integral, uint8_t, true);
CVMUTIL_DECLARE_TRAITS(is_integral, uint16_t, true);
CVMUTIL_DECLARE_TRAITS(is_integral, uint32_t, true);
CVMUTIL_DECLARE_TRAITS(is_integral, uint64_t, true);

CVMUTIL_DECLARE_TRAITS(is_floating_point, float, true);
CVMUTIL_DECLARE_TRAITS(is_floating_point, double, true);

#endif

CVMUTIL_DECLARE_TYPE_NAME(float, "float");
CVMUTIL_DECLARE_TYPE_NAME(double, "double");
CVMUTIL_DECLARE_TYPE_NAME(int, "int");
CVMUTIL_DECLARE_TYPE_NAME(uint32_t, "int (non-negative)");
CVMUTIL_DECLARE_TYPE_NAME(uint64_t, "long (non-negative)");
CVMUTIL_DECLARE_TYPE_NAME(std::string, "string");
CVMUTIL_DECLARE_TYPE_NAME(bool, "boolean");
CVMUTIL_DECLARE_TYPE_NAME(void*, "ptr");

template<typename Then, typename Else>
struct IfThenElseType<true, Then, Else> {
  typedef Then Type;
};

template<typename Then, typename Else>
struct IfThenElseType<false, Then, Else> {
  typedef Else Type;
};
//! \endcond
}  // namespace utils
#endif  // CVMUTIL_TYPE_TRAITS_H_
