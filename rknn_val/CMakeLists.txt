cmake_minimum_required(VERSION 3.10)
project(rknn_val)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils/ utils.out)

include_directories(${PROJECT_SOURCE_DIR}/3rdparty/rknpu2/include)

# add_library(static_lib ${PROJECT_SOURCE_DIR}/static_src)

add_executable(rknn_val
    src/val.cc
    src/postprocess.cc
    src/yolov8.cc
)

set(LIBRKNNRT
    ${PROJECT_SOURCE_DIR}/3rdparty/rknpu2/Linux/aarch64/librknnrt.so
)

target_link_libraries(rknn_val
    dl
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNRT}
)

add_executable(rknn_detect
    src/detect.cc
    src/postprocess.cc
    src/yolov8.cc
)

target_link_libraries(rknn_detect
    dl
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNRT}
)