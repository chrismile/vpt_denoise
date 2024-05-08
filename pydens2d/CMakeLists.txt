cmake_minimum_required(VERSION 3.11)
project(PyLimbo)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCES src/*.cpp src/*.c src/*.h src/*.hpp)
file(GLOB_RECURSE CUDA_SOURCES src/*.cu src/*.cuh)
list(REMOVE_ITEM CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/MutualInformationKraskov.cu)

include_directories(src)

if(MSVC)
    set(CMAKE_CXX_FLAGS "-W3 /EHsc")
elseif(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-Wall")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-Wall")
endif()

if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    # https://cmake.org/cmake/help/git-stage/module/FindCUDA.html
    find_package(CUDA QUIET)
    if(${CUDA_FOUND})
        add_library(pylimbo SHARED ${SOURCES} ${CUDA_SOURCES})
    else()
        add_library(pylimbo SHARED ${SOURCES})
    endif()
else()
    # https://cmake.org/cmake/help/git-stage/module/FindCUDAToolkit.html
    find_package(CUDAToolkit QUIET)
    if(${CUDAToolkit_FOUND})
        add_library(pylimbo SHARED ${SOURCES} ${CUDA_SOURCES})
    else()
        add_library(pylimbo SHARED ${SOURCES})
    endif()
endif()

cmake_policy(SET CMP0012 NEW)
find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)

find_package(Python3 COMPONENTS Development REQUIRED)
target_link_libraries(pylimbo PRIVATE Python3::Python)

target_link_libraries(pylimbo PRIVATE ${TORCH_LIBRARIES})
target_include_directories(pylimbo PUBLIC ${TORCH_INCLUDE_DIR})

target_include_directories(pylimbo PUBLIC third_party/limbo/src)
target_include_directories(pylimbo PUBLIC third_party/eigen)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

install(
        TARGETS pylimbo
        ARCHIVE DESTINATION modules
        LIBRARY DESTINATION modules
        RUNTIME DESTINATION bin
)
