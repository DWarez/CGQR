cmake_minimum_required(VERSION 3.21)
project(CGQR)

set(CMAKE_CXX_STANDARD 20)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

add_executable(CGQR src/main.cpp src/main.cpp include/conjugate_gradient.hpp)

target_link_libraries(CGQR ${ARMADILLO_LIBRARIES})