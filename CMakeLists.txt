cmake_minimum_required(VERSION 3.10)

project(foolish_trash VERSION 22.6.30 LANGUAGES CXX)

# Specialize CMake build flags for different build types
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-deprecated")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Don't build test by default
set(BUILD_TEST False)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Add include dir
include_directories(
    ${PROJECT_SOURCE_DIR}/devices
    ${PROJECT_SOURCE_DIR}/modules
)

add_executable(foolish_trash 
    ${PROJECT_SOURCE_DIR}/src/main.cpp
)

# Dependence lib
## Find and add librealsense2 as required
find_package(realsense2 REQUIRED)
include_directories(${realsense2_INCLUDE_DIRS})
link_libraries(${realsense2_LIBRARY})

## Find and add opencv as required
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

## Find and add apriltag as required
find_package(apriltag REQUIRED)
include_directories(${apriltag_INCLUDE_DIRS} ${apriltag_INCLUDE_DIRS}/apriltag)
link_directories(${apriltag_LIBRARY})

# Add Subdirectories
add_subdirectory(${PROJECT_SOURCE_DIR}/devices/intel_rs)
add_subdirectory(${PROJECT_SOURCE_DIR}/modules/solve_pose)
add_subdirectory(${PROJECT_SOURCE_DIR}/modules/april_tag)

# Target
target_link_libraries(foolish_trash 
    ${OpenCV_LIBS}
    ${realsense2_LIBRARY}
    ${apriltag_LIBRARY}
    intel_camera
    solve_pose
    april_tag
)



