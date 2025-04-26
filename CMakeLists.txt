cmake_minimum_required(VERSION 3.15)

project(ttMandelbrot)

set(CMAKE_CXX_STANDARD 20)

find_package(PNG REQUIRED)
add_library(utils STATIC
    utils/stb_image_write.c
)
target_include_directories(utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)
target_link_libraries(utils PRIVATE PNG::PNG)

find_package(OpenMP REQUIRED)
add_executable(cpu cpu.cpp)
target_link_libraries(cpu PRIVATE utils OpenMP::OpenMP_CXX)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(CPM)

if(CMAKE_VERSION VERSION_LESS 3.25)
    add_subdirectory(dependencies EXCLUDE_FROM_ALL)
else()
    add_subdirectory(dependencies EXCLUDE_FROM_ALL SYSTEM)
endif()

if("$ENV{TT_METAL_HOME}" STREQUAL "")
    message(FATAL_ERROR "TT_METAL_HOME is not set")
endif()

add_library(ttmetal IMPORTED INTERFACE)
target_include_directories(ttmetal INTERFACE
    $ENV{TT_METAL_HOME}
    $ENV{TT_METAL_HOME}/tt_metal
    $ENV{TT_METAL_HOME}/tt_metal/api/
    $ENV{TT_METAL_HOME}/tt_metal/include
    $ENV{TT_METAL_HOME}/tt_metal/hostdevcommon/api
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/api/
    $ENV{TT_METAL_HOME}/tt_metal/third_party/tracy/public/
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/
    $ENV{TT_METAL_HOME}/tt_stl/
)

target_link_directories(ttmetal INTERFACE
    $ENV{TT_METAL_HOME}/build/tt_metal
)

target_link_libraries(ttmetal INTERFACE
    fmt
    magic_enum
    Reflect::Reflect
    yaml-cpp
    Boost::core
    Boost::container
    tt_metal
)

add_executable(tt_single_core single_core/tt_single_core.cpp)
target_link_libraries(tt_single_core
    PRIVATE
    ttmetal
    utils
    OpenMP::OpenMP_CXX
)

add_executable(tt_single_core_nullary single_core_nullary/tt_single_core_nullary.cpp)
target_link_libraries(tt_single_core_nullary
    PRIVATE
    ttmetal
    utils
    OpenMP::OpenMP_CXX
)

add_executable(tt_multi_core_nullary multi_core_nullary/tt_multi_core_nullary.cpp)
target_link_libraries(tt_multi_core_nullary
    PRIVATE
    ttmetal
    utils
    OpenMP::OpenMP_CXX
)
