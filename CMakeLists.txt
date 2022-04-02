cmake_minimum_required(VERSION 3.21)
project(CGQR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS -O3)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

add_executable(CGQR src/main.cpp include/conjugate_gradient.hpp include/qr_factorization.hpp include/utils.hpp src/utils.cpp)

add_executable(qr_test tests/qr_test.cpp include/qr_factorization.hpp src/qr_factorization.cpp include/utils.hpp src/utils.cpp)
add_executable(cg_test tests/cg_test.cpp include/conjugate_gradient.hpp src/conjugate_gradient.cpp include/utils.hpp src/utils.cpp)

target_link_libraries(CGQR ${ARMADILLO_LIBRARIES})
target_link_libraries(qr_test ${ARMADILLO_LIBRARIES})
target_link_libraries(cg_test ${ARMADILLO_LIBRARIES})