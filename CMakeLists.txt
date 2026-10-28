cmake_minimum_required(VERSION 3.20)
project(w4a16_gemv_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(w4a16_bench
    src/reference.cpp
    src/candidate.cpp
    bench/harness.cpp
)

target_include_directories(w4a16_bench PRIVATE include)

if (MSVC)
    target_compile_options(w4a16_bench PRIVATE /O2 /arch:AVX2 /openmp)
else()
    target_compile_options(w4a16_bench PRIVATE -O3 -mavx2 -mfma -Wall -Wextra)
    target_link_libraries(w4a16_bench PRIVATE OpenMP::OpenMP_CXX)
endif()
