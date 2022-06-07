cmake_minimum_required(VERSION 3.16)
project(CGQR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3")

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

add_executable(CGQR src/main.cpp src/conjugate_gradient.cpp src/qr_factorization.cpp src/utils.cpp)
add_executable(qr_test tests/qr_test.cpp src/qr_factorization.cpp src/utils.cpp)
add_executable(cg_test tests/cg_test.cpp src/conjugate_gradient.cpp src/utils.cpp)

target_link_libraries(CGQR ${ARMADILLO_LIBRARIES})
target_link_libraries(qr_test ${ARMADILLO_LIBRARIES})
target_link_libraries(cg_test ${ARMADILLO_LIBRARIES})
