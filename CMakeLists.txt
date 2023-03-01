cmake_minimum_required(VERSION 3.24)
project(numeric_methods_mss)

set(CMAKE_CXX_STANDARD 17)

add_executable(Laplacian_test main_lap.cpp
        progonka/thomas.cpp
        progonka/thomas.h
        diffusion_practise_one/Laplacian.h
        diffusion_practise_one/Laplacian.cpp)

add_executable(thomas_test main.cpp
        progonka/thomas.cpp
        progonka/thomas.h)