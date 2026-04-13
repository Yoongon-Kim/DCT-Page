cmake_minimum_required(VERSION 3.26.4)

# Use system default compilers (override with -DCMAKE_C_COMPILER / -DCMAKE_CXX_COMPILER)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# QUEST_KERNELS_DIR must point to the Quest repo's kernels/ directory.
# Set via: cmake -DQUEST_KERNELS_DIR=/path/to/Quest/kernels ...
if(NOT DEFINED QUEST_KERNELS_DIR)
    message(FATAL_ERROR "QUEST_KERNELS_DIR is not set. Point it to the Quest repo's kernels/ directory.")
endif()

# ------------- configure rapids-cmake --------------#
include(${CMAKE_SOURCE_DIR}/cmake/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

project(_kernels LANGUAGES CUDA CXX)

# ------------- configure raft -----------------#
rapids_cpm_init()
include(${CMAKE_SOURCE_DIR}/cmake/get_raft.cmake)

# Check: https://stackoverflow.com/questions/68401650/how-can-i-make-a-pytorch-extension-with-cmake
# Fix linking error: https://github.com/pytorch/pytorch/issues/108041
find_package(Python REQUIRED COMPONENTS Interpreter Development)

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

# ------------- pybind11 -----------------------#
# Use the submodule if present, otherwise fetch from GitHub
set(_PYBIND_DIR "${QUEST_KERNELS_DIR}/3rdparty/pybind")
if(EXISTS "${_PYBIND_DIR}/CMakeLists.txt")
    add_subdirectory(${_PYBIND_DIR} ${CMAKE_BINARY_DIR}/pybind11)
else()
    message(STATUS "pybind11 submodule not found at ${_PYBIND_DIR}, fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

file(GLOB PYTORCH_SOURCES "csrc/*.cu")
pybind11_add_module(_kernels MODULE ${PYTORCH_CPP_SOURCES} ${PYTORCH_SOURCES})

target_compile_definitions(_kernels PRIVATE -DBSK_TORCH_CHECK) # Enable Torch Tensor Dimension Check
target_include_directories(_kernels PRIVATE ${QUEST_KERNELS_DIR}/include)
target_include_directories(_kernels PRIVATE ${QUEST_KERNELS_DIR}/3rdparty/flashinfer/include)
target_compile_options(_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
target_link_libraries(_kernels PRIVATE ${TORCH_LIBRARIES} raft::raft Python::Python pybind11::module ${TORCH_PYTHON_LIBRARY})
