cmake_minimum_required (VERSION 2.8)

project (kernel_lib)

find_package(CUDA)

add_compile_options(-MT)
set(CMAKE_BUILD_TYPE Release)
file(GLOB_RECURSE  KERNEL_SRC ./src/*.cu)

include_directories(./inc)

cuda_add_library(kernel_lib STATIC ${KERNEL_SRC})