aux_source_directory(. DIR_APR_TGA_SRCS)
add_library(april_tag ${DIR_APR_TGA_SRCS})

## Find and add apriltag as required
find_package(PkgConfig)
pkg_search_module(apriltag REQUIRED apriltag)
set(apriltag_INCLUDE_DIRS "${apriltag_INCLUDE_DIRS}/apriltag")
link_directories(${apriltag_LIBDIR})
include_directories(${apriltag_INCLUDE_DIRS})
# message(STATUS "${apriltag_INCLUDE_DIRS}")
target_link_libraries(april_tag ${apriltag_LIBRARIES})