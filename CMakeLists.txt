cmake_minimum_required(VERSION 3.26.5)
project(sequential CUDA)

set(CMAKE_CXX_STANDARD 20)

add_executable(sequential main.cu
        Models.h
)
