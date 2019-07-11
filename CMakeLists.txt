cmake_minimum_required(VERSION 3.2)
project(cvm C CXX)

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
# Utility functions
include(cmake/util/Util.cmake)
include(cmake/util/FindCUDA.cmake)
include(cmake/util/FindOpenmp.cmake)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

# NOTE: do not modify this file to change option values.
# You can create a config.cmake at build folder
# and add set(OPTION VALUE) to override these build options.
# Alernatively, use cmake -DOPTION=VALUE through command-line.
cvm_option(USE_CUDA "Build with CUDA" ON)
cvm_option(USE_LLVM "Build with LLVM, can be set to specific llvm-config path" ON)
cvm_option(USE_CVM_RUNTIME "Build with tiny cvm runtime" ON)
cvm_option(USE_RTTI "Build with RTTI" ON)
cvm_option(INSTALL_DEV "Install compiler infrastructure" OFF)

# include directories
include_directories("include")

# initial variables
set(CVM_LINKER_LIBS "")
set(CVM_RUNTIME_LINKER_LIBS "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Generic compilation options
if(MSVC)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
  add_definitions(-DHalide_SHARED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")
  if(USE_MSVC_MT)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()
else(MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++11"    SUPPORT_CXX11)
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_C_FLAGS "-mavx2 -Og -g -Wall -fPIC ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-mavx2 -Og -g -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_C_FLAGS "-mavx2 -O3 -Wall -fPIC ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-mavx2 -O3 -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")
  endif ()
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(CMAKE_CXX_FLAGS "-faligned-new ${CMAKE_CXX_FLAGS}")
  endif()
endif(MSVC)

# add source group
FILE(GLOB_RECURSE GROUP_INCLUDE "src/*.h" "include/*.h")
assign_source_group("Include" ${GROUP_INCLUDE})

# Source file lists
file(GLOB RUNTIME_SRCS src/*.cc)

# Package runtime rules
if(NOT USE_RTTI)
  add_definitions(-DDMLC_ENABLE_RTTI=0)
endif()

message(STATUS "Build with CVM runtime support...")
file(GLOB RUNTIME_CVM_SRCS src/cvm/*.cc)
if(${USE_CUDA} STREQUAL "ON")
    message("use cuda")
    project(cvm CUDA)
    set (EXECUTE_NAME cvm_runtime_cuda)
    add_definitions(-DCVM_RUNTIME_CUDA)
    file(GLOB RUNTIME_CVM_CUDA_SRCS src/cuda/*.cc)
    file(GLOB CVM_OPS_CUDA_SRCS src/cvm/ops/cuda/*.cc
      src/cvm/ops/cuda/*.cu)
    list(APPEND RUNTIME_CVM_SRCS ${CVM_OPS_CUDA_SRCS})
    list(APPEND RUNTIME_SRCS ${RUNTIME_CVM_CUDA_SRCS})
else()
    file(GLOB CVM_OPS_CPU_SRCS src/cvm/ops/cpu/*.cc)
    list(APPEND RUNTIME_CVM_SRCS ${CVM_OPS_CPU_SRCS})
    set (EXECUTE_NAME cvm_runtime_cpu)
endif()
list(APPEND RUNTIME_SRCS ${RUNTIME_CVM_SRCS})
file(GLOB CVM_TOP_SRCS src/cvm/top/*.cc
          src/cvm/top/nn/*.cc
          src/cvm/top/tensor/*.cc)
list(APPEND RUNTIME_SRCS ${CVM_TOP_SRCS})

# Module rules
include(cmake/modules/CUDA.cmake)

add_library(${EXECUTE_NAME} SHARED ${RUNTIME_SRCS})
target_link_libraries(${EXECUTE_NAME} ${CVM_RUNTIME_LINKER_LIBS})

# Installation rules
install(TARGETS ${EXECUTE_NAME} DESTINATION lib${LIB_SUFFIX})

# More target definitions
if(MSVC)
    target_compile_definitions(${EXECUTE_NAME} PRIVATE -DHalide_EXPORTS)
    target_compile_definitions(${EXECUTE_NAME} PRIVATE -DCVM_EXPORTS)
endif()
