cmake_minimum_required (VERSION 2.8)

project(kernel_test)

#vcvarsall.bat

aux_source_directory(./test DIR_SRC)
add_subdirectory(./kernel)
include_directories(./kernel/inc
                    ./test
                    ../3rdparty/inc)
LINK_DIRECTORIES(../3rdparty/libs)

if(MSVC)     
    # Use the static C library for all build types
    foreach(var 
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
      )
      if(${var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${var} "${${var}}")
      endif()
    endforeach()    
endif(MSVC)
set(CMAKE_BUILD_TYPE Release)

add_executable(kernel_test ${DIR_SRC})

target_link_libraries(kernel_test kernel_lib 
                                  ade.lib
                                  IlmImf.lib
                                  ippicvmt.lib
                                  ippiw.lib
                                  ittnotify.lib
                                  libjpeg-turbo.lib
                                  libopenjp2.lib
                                  libpng.lib
                                  libprotobuf.lib
                                  libtiff.lib
                                  libwebp.lib
                                  opencv_calib3d452.lib
                                  opencv_core452.lib
                                  opencv_dnn452.lib
                                  opencv_features2d452.lib
                                  opencv_flann452.lib
                                  opencv_gapi452.lib
                                  opencv_highgui452.lib
                                  opencv_imgcodecs452.lib
                                  opencv_imgproc452.lib
                                  opencv_ml452.lib
                                  opencv_objdetect452.lib
                                  opencv_photo452.lib
                                  opencv_stitching452.lib
                                  opencv_ts452.lib
                                  opencv_video452.lib
                                  opencv_videoio452.lib
                                  quirc.lib
                                  zlib.lib)
