cmake_minimum_required(VERSION 3.16)
project(srm_extract LANGUAGES CXX)

add_executable(srm_extract
    portable_main.cpp
    SRM_src/SRMclass.cpp
    SRM_src/submodel.cpp
    SRM_src/s.cpp
    SRM_src/image.cpp
)
target_include_directories(srm_extract PRIVATE SRM_src)
target_compile_features(srm_extract PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(srm_extract PRIVATE /O2 /W3)
else()
    target_compile_options(srm_extract PRIVATE -O3 -Wall -Wextra -Wno-unused-parameter)
endif()
