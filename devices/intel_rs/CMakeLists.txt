aux_source_directory(. DIR_INTEL_RS_SRCS)
add_library(intel_camera ${DIR_INTEL_RS_SRCS})

## Find and add librealsense2 as required
find_package(realsense2 REQUIRED)
include_directories(${realsense2_INCLUDE_DIRS})
link_libraries(${realsense2_LIBRARY})
target_link_libraries(intel_camera ${realsense2_LIBRARY})