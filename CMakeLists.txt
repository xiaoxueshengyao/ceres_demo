cmake_minimum_required(VERSION 2.8)
project(ceres_demon)

set(CMAKE_CXX_STANDARD 14)

# find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
    ${CERES_INCLUDE_DIRS}
)

add_executable(helloworld src/helloworld.cpp)
target_link_libraries(helloworld ${CERES_LIBRARIES})
