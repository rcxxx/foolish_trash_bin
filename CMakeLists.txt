cmake_minimum_required(VERSION 3.10)

project(foolish_trash VERSION 22.6.30 LANGUAGES CXX)

# Specialize CMake build flags for different build types
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-deprecated")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Don't build test by default
set(BUILD_TEST False)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Add include dir
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/devices
    ${PROJECT_SOURCE_DIR}/modules
)

add_executable(foolish_trash 
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/include/position.hpp
)

# Dependence lib

## Find and add opencv as required
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

## Add onnxruntime
set(OpenVINO_DIR ~/intel/openvino_2022/runtime/cmake)
find_package(OpenVINO REQUIRED)
include_directories(
    ~/intel/openvino_2022/runtime/include
    ~/intel/openvino_2022/runtime/include/ie)
link_libraries(openvino::runtime)

# Add Subdirectories
add_subdirectory(${PROJECT_SOURCE_DIR}/devices/intel_rs)
add_subdirectory(${PROJECT_SOURCE_DIR}/modules/solve_pose)
add_subdirectory(${PROJECT_SOURCE_DIR}/modules/april_tag)
add_subdirectory(${PROJECT_SOURCE_DIR}/modules/yolov5)

# Target
target_link_libraries(foolish_trash 
    ${OpenCV_LIBS}
    openvino::runtime
    intel_camera
    solve_pose
    april_tag
    yolov5
)
