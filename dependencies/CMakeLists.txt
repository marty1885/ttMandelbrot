# Shadow the cache variable with a blank value
# Placing a no-op .clang-tidy file at the root of CPM cache is insufficient as some projects may define
# their own .clang-tidy within themselves and still not be clean against it <cough>flatbuffers</cough>
set(CMAKE_C_CLANG_TIDY "")
set(CMAKE_CXX_CLANG_TIDY "")

############################################################################################################################
# Boost
############################################################################################################################

include(${PROJECT_SOURCE_DIR}/cmake/fetch_boost.cmake)

fetch_boost_library(core)
fetch_boost_library(smart_ptr)
fetch_boost_library(container)

add_library(span INTERFACE)
target_link_libraries(span INTERFACE Boost::core)

############################################################################################################################
# yaml-cpp
############################################################################################################################

CPMAddPackage(
    NAME yaml-cpp
    GITHUB_REPOSITORY jbeder/yaml-cpp
    GIT_TAG 0.8.0
    PATCHES
        yaml-cpp.patch
    OPTIONS
        "YAML_CPP_BUILD_TESTS OFF"
        "YAML_CPP_BUILD_TOOLS OFF"
        "YAML_BUILD_SHARED_LIBS OFF"
)

if(yaml-cpp_ADDED)
    set_target_properties(
        yaml-cpp
        PROPERTIES
            DEBUG_POSTFIX
                ""
    )
endif()

############################################################################################################################
# boost-ext reflect : https://github.com/boost-ext/reflect
############################################################################################################################

CPMAddPackage(NAME reflect GITHUB_REPOSITORY boost-ext/reflect GIT_TAG v1.1.1)
if(reflect_ADDED)
    add_library(reflect INTERFACE)
    add_library(Reflect::Reflect ALIAS reflect)
    target_include_directories(reflect SYSTEM INTERFACE ${reflect_SOURCE_DIR})
endif()

############################################################################################################################
# magic_enum : https://github.com/Neargye/magic_enum
############################################################################################################################

CPMAddPackage(NAME magic_enum GITHUB_REPOSITORY Neargye/magic_enum GIT_TAG v0.9.7)

############################################################################################################################
# fmt : https://github.com/fmtlib/fmt
############################################################################################################################

CPMAddPackage(NAME fmt GITHUB_REPOSITORY fmtlib/fmt GIT_TAG 11.0.1)

############################################################################################################################
# range-v3 : https://github.com/ericniebler/range-v3
############################################################################################################################

CPMAddPackage(NAME range-v3 GITHUB_REPOSITORY ericniebler/range-v3 PATCHES ranges-v3.patch GIT_TAG 0.12.0)

############################################################################################################################
# nlohmann/json : https://github.com/nlohmann/json
############################################################################################################################

CPMAddPackage(NAME json GITHUB_REPOSITORY nlohmann/json GIT_TAG v3.12.0)
