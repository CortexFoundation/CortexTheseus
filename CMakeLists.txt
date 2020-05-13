cmake_minimum_required(VERSION 3.2)
project(cvm C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON )

# Utility functions
include(cmake/util/Util.cmake)
include(cmake/util/FindCUDA.cmake)
include(cmake/util/FindOpenmp.cmake)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
endif()

# include directories
include_directories("include")

# Generic compilation options
set(CMAKE_C_FLAGS "-fsigned-char -Wall -fPIC ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-fsigned-char -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")
if (DEBUG)
  set(CMAKE_C_FLAGS "-Og -g ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-Og -g ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_C_FLAGS "-O3 ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
endif ()

if(PROFILE)
  add_definitions(-DPROFILE)
  if (ENABLE_CUDA)
    add_definitions(-DCUDA_PROFILE)
  endif()
endif()

# compile source files
set (CVM_RUNTIME_LINKER_LIBS "")
set (EXECUTE_NAME cvm_runtime)
set (LIBRARY_NAME cvm)

# Source file lists
file(GLOB RUNTIME_SRCS src/core/*.cc
                       src/runtime/*.cc
                       src/runtime/cpu/devapi/*.cc)

file(GLOB CVM_SRCS src/compiler/*.cc
                   src/pass/*.cc)

if(ENABLE_CPU)
    message(STATUS "ENABLE CPU")
    file(GLOB CVM_OPS_CPU_SRCS src/runtime/cpu/ops/*.cc)
    list(APPEND RUNTIME_SRCS ${CVM_OPS_CPU_SRCS})
endif()

if(ENABLE_FORMAL)
    message(STATUS "ENABLE FORMALIZATION")
    file(GLOB RUNTIME_FORMAL_SRCS src/runtime/formal/devapi/*.cc)
    list(APPEND RUNTIME_SRCS ${RUNTIME_FORMAL_SRCS})

    file(GLOB CVM_OPS_FORMAL_SRCS src/runtime/formal/ops/*.cc)
    list(APPEND RUNTIME_SRCS ${CVM_OPS_FORMAL_SRCS})
endif()

if(ENABLE_CUDA)
    message(STATUS "ENABLE CUDA")
    # project(cvm CUDA)

    find_cuda(${ENABLE_CUDA})
    if (CUDA_FOUND)
      include_directories(${CUDA_INCLUDE_DIRS})
    else()
      message(FATAL_ERROR "Cannot find CUDA, ENABLE_CUDA=" ${ENABLE_CUDA})
    endif()

    list(APPEND CVM_RUNTIME_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
    list(APPEND CVM_RUNTIME_LINKER_LIBS ${CUDA_CUDA_LIBRARY})
    list(APPEND CVM_RUNTIME_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --expt-extended-lambda)

    add_definitions(-DCVM_RUNTIME_CUDA)
    if(USE_NANO)
      add_definitions(-DNANO)
    endif()
      
    file(GLOB RUNTIME_CUDA_SRCS src/runtime/cuda/devapi/*.cc)
    list(APPEND RUNTIME_SRCS ${RUNTIME_CUDA_SRCS})

    file(GLOB CVM_OPS_CUDA_SRCS src/runtime/cuda/ops/*.cc)
    list(APPEND RUNTIME_SRCS ${CVM_OPS_CUDA_SRCS})

    file(GLOB CUDA_SRCS src/runtime/cuda/ops/*.cu)
    cuda_compile(CUDA_LIBS ${CUDA_SRCS})
    list(APPEND RUNTIME_SRCS ${CUDA_LIBS})
endif()

if(ENABLE_OPENCL)
  message(STATUS "ENABLE OPENCL")

# list(APPEND RUNTIME_SRCS ${RUNTIME_CUDA_SRCS})
#  list(APPEND CVM_RUNTIME_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
#  list(APPEND CVM_RUNTIME_LINKER_LIBS ${CUDA_CUDA_LIBRARY})

  set(CMAKE_C_FLAGS "-lOpenCL ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-lOpenCL ${CMAKE_CXX_FLAGS}")

  file(GLOB RUNTIME_OPENCL_SRCS src/runtime/opencl/devapi/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_OPENCL_SRCS})

  file(GLOB CVM_OPS_OPENCL_SRCS src/runtime/opencl/ops/*.cc)
  list(APPEND RUNTIME_SRCS ${CVM_OPS_OPENCL_SRCS})

endif()

list(APPEND RUNTIME_SRCS ${CVM_SRCS})

file(GLOB CVM_TOP_SRCS src/top/*.cc
                       src/top/nn/*.cc
                       src/top/tensor/*.cc)
list(APPEND RUNTIME_SRCS ${CVM_TOP_SRCS})

add_library(${EXECUTE_NAME} SHARED ${RUNTIME_SRCS})
target_link_libraries(${EXECUTE_NAME} ${CVM_RUNTIME_LINKER_LIBS})

list(APPEND CVM_SRCS ${RUNTIME_SRCS})
add_library(${LIBRARY_NAME} SHARED ${CVM_SRCS})
target_link_libraries(${LIBRARY_NAME} ${CVM_RUNTIME_LINKER_LIBS})


