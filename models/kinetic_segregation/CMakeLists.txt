cmake_minimum_required(VERSION 3.20)

# Enable Objective-C only on Apple (for Metal backend).
if(APPLE)
    project(kinetic_segregation C CXX OBJC)
else()
    project(kinetic_segregation C CXX)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build (-O3) if not specified.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Put the binary in the project root (where tests expect it).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# ── Core simulation library (portable C99) ──────────────────────────
add_library(ks_core STATIC
    src/simulation.c
    src/potentials.c
    src/cell_list.c
    src/rng.c
)
target_include_directories(ks_core PUBLIC src)
if(UNIX AND NOT APPLE)
    target_link_libraries(ks_core PRIVATE m)
endif()

# ── GPU backend ─────────────────────────────────────────────────────
if(APPLE)
    # Metal backend (macOS / Apple Silicon).
    add_library(ks_gpu_backend STATIC src/metal_engine.m)
    target_include_directories(ks_gpu_backend PRIVATE src)
    target_compile_options(ks_gpu_backend PRIVATE -fobjc-arc)

    # Pre-compile Metal shaders if the toolchain is available.
    # Note: xcrun --find may succeed even if the toolchain is not installed,
    # so we test by actually invoking the compiler with --version.
    execute_process(
        COMMAND xcrun -sdk macosx metal --version
        RESULT_VARIABLE _metal_result
        OUTPUT_QUIET ERROR_QUIET
    )
    if(_metal_result EQUAL 0)
        add_custom_command(
            OUTPUT ${CMAKE_SOURCE_DIR}/src/shaders.metallib
            COMMAND xcrun -sdk macosx metal -c
                -I${CMAKE_SOURCE_DIR}/src
                -o ${CMAKE_BINARY_DIR}/shaders.air
                ${CMAKE_SOURCE_DIR}/src/shaders.metal
            COMMAND xcrun -sdk macosx metallib
                -o ${CMAKE_SOURCE_DIR}/src/shaders.metallib
                ${CMAKE_BINARY_DIR}/shaders.air
            DEPENDS
                ${CMAKE_SOURCE_DIR}/src/shaders.metal
                ${CMAKE_SOURCE_DIR}/src/ks_physics.h
            COMMENT "Pre-compiling Metal shaders"
        )
        add_custom_target(metallib ALL
            DEPENDS ${CMAKE_SOURCE_DIR}/src/shaders.metallib
        )
    else()
        add_custom_target(metallib
            COMMAND ${CMAKE_COMMAND} -E echo
                "Metal compiler not found -- shaders will compile at runtime"
        )
    endif()
else()
    # Stub backend (returns NULL → CPU fallback).
    # Future: detect CUDA and use cuda_engine.cu instead.
    add_library(ks_gpu_backend STATIC src/gpu_stub.c)
    target_include_directories(ks_gpu_backend PRIVATE src)

    add_custom_target(metallib
        COMMAND ${CMAKE_COMMAND} -E echo "Metal not available on this platform"
    )
endif()

# ── Main executable ─────────────────────────────────────────────────
add_executable(ks_gpu src/main.cpp)
target_include_directories(ks_gpu PRIVATE src)
target_link_libraries(ks_gpu PRIVATE ks_core ks_gpu_backend)
if(APPLE)
    target_link_libraries(ks_gpu PRIVATE
        "-framework Metal"
        "-framework Foundation"
    )
endif()

# ── Test shared library (Python ctypes) ─────────────────────────────
option(BUILD_TESTLIB "Build shared library for ctypes tests" ON)
if(BUILD_TESTLIB)
    add_library(ks_potentials SHARED src/potentials.c src/cell_list.c src/rng.c)
    target_include_directories(ks_potentials PRIVATE src)
    if(UNIX AND NOT APPLE)
        target_link_libraries(ks_potentials PRIVATE m)
    endif()
    set_target_properties(ks_potentials PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

# ── Optional PDF documentation ──────────────────────────────────────
# Prefer tectonic (single-pass, auto-downloads packages); fall back to pdflatex.
find_program(TECTONIC tectonic)
find_program(PDFLATEX pdflatex)
if(TECTONIC)
    add_custom_target(pdf
        COMMAND ${TECTONIC} methods.tex
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/Methods
        COMMENT "Compiling methods.tex → methods.pdf (tectonic)"
    )
elseif(PDFLATEX)
    add_custom_target(pdf
        COMMAND ${PDFLATEX} -interaction=nonstopmode methods.tex
        COMMAND ${PDFLATEX} -interaction=nonstopmode methods.tex
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/Methods
        COMMENT "Compiling methods.tex → methods.pdf (pdflatex)"
    )
else()
    add_custom_target(pdf
        COMMAND ${CMAKE_COMMAND} -E echo
            "No TeX compiler found -- install tectonic or pdflatex to build PDF"
    )
endif()
