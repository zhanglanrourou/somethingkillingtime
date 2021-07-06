cmake_minimum_required (VERSION 2.8)

project(kernel_test)

#vcvarsall.bat

aux_source_directory(./test DIR_SRC)
add_subdirectory(./kernel)
include_directories(./kernel/inc
                    ./test
                    ../3rdparty/opencv)

add_compile_options(-MT)
set(CMAKE_BUILD_TYPE Release)
set(TEST_DATA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/test/data)
configure_file("./test/data_path.h.in"
               "${CMAKE_CURRENT_SOURCE_DIR}/test/data_path.h")
add_executable(kernel_test ${DIR_SRC})

target_link_libraries(kernel_test kernel_lib)
target_link_libraries(kernel_test opencv_core
                                  opencv_highgui
                                  opencv_imgcodecs
                                  opencv_imgproc)
