cmake_minimum_required(VERSION 3.22)

project(MCPA VERSION 0.1.0 LANGUAGES CXX CUDA)

include(CTest)
find_package(CUDAToolkit 12.4 REQUIRED)

option(MCPA_BUILD_TESTS "Build the modular CPU and CUDA test suites" ON)
option(MCPA_BUILD_BENCHMARKS "Build the equilibration benchmark" OFF)
option(MCPA_BW_GROUPED_ATTEMPTS
    "Process each Baxter-Wu curand4 result as one explicit ordered attempt group"
    OFF)
option(MCPA_BW_NEIGHBOR_TABLE
    "Use an opt-in precomputed SLF neighbor table in Baxter-Wu equilibration"
    OFF)

if(MCPA_BW_GROUPED_ATTEMPTS AND MCPA_BW_NEIGHBOR_TABLE)
    message(FATAL_ERROR
        "Grouped attempts and the neighbor table are isolated experiments; enable only one")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# GCC 8 implements C++17 std::filesystem in a separate compatibility library.
# Newer GCC releases fold it into libstdc++.
set(MCPA_FILESYSTEM_COMPAT_LIBRARY "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    set(MCPA_FILESYSTEM_COMPAT_LIBRARY stdc++fs)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures" FORCE)
endif()

add_library(mcpa_baxterwu STATIC
    lib/baxterwu_lib.cu
    lib/checkpoint.cpp
)
target_include_directories(mcpa_baxterwu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/lib")
target_link_libraries(mcpa_baxterwu PUBLIC CUDA::cudart)
if(MCPA_BW_GROUPED_ATTEMPTS)
    target_compile_definitions(mcpa_baxterwu PRIVATE MCPA_BW_GROUPED_ATTEMPTS=1)
endif()
if(MCPA_BW_NEIGHBOR_TABLE)
    target_compile_definitions(mcpa_baxterwu PUBLIC MCPA_BW_NEIGHBOR_TABLE=1)
endif()
target_compile_options(mcpa_baxterwu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra;-Xptxas=-warn-spills>
)
set_target_properties(mcpa_baxterwu PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_executable(main_bw main/main.cpp)
target_link_libraries(main_bw PRIVATE
    mcpa_baxterwu
    ${MCPA_FILESYSTEM_COMPAT_LIBRARY}
)

if(BUILD_TESTING AND MCPA_BUILD_TESTS)
    # Add a focused test by creating tests/test_cpu_<topic>.cpp or
    # tests/test_cuda_<topic>.cu; CMake notices new files automatically.
    file(GLOB MCPA_CPU_ALL_TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpu*.cpp"
    )
    file(GLOB MCPA_CUDA_ALL_TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda*.cu"
    )
    set(MCPA_CPU_CHECKPOINT_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpu_checkpoint.cpp")
    set(MCPA_CUDA_CHECKPOINT_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_checkpoint.cu")
    set(MCPA_CPU_TEST_SOURCES ${MCPA_CPU_ALL_TEST_SOURCES})
    set(MCPA_CUDA_TEST_SOURCES ${MCPA_CUDA_ALL_TEST_SOURCES})
    list(REMOVE_ITEM MCPA_CPU_TEST_SOURCES ${MCPA_CPU_CHECKPOINT_SOURCE})
    list(REMOVE_ITEM MCPA_CUDA_TEST_SOURCES ${MCPA_CUDA_CHECKPOINT_SOURCE})
    add_executable(mcpa_cpu_tests
        tests/test_main.cpp
        ${MCPA_CPU_ALL_TEST_SOURCES}
    )
    target_include_directories(mcpa_cpu_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    )
    target_link_libraries(mcpa_cpu_tests PRIVATE
        mcpa_baxterwu
        ${MCPA_FILESYSTEM_COMPAT_LIBRARY}
    )

    add_executable(mcpa_cuda_tests
        tests/test_cuda_main.cpp
        ${MCPA_CUDA_ALL_TEST_SOURCES}
    )
    target_include_directories(mcpa_cuda_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    )
    target_link_libraries(mcpa_cuda_tests PRIVATE
        mcpa_baxterwu
        ${MCPA_FILESYSTEM_COMPAT_LIBRARY}
    )

    # Register every doctest TEST_CASE as its own CTest entity.  This keeps the
    # command line output human-readable and ensures a failure names the exact
    # contract that broke.  Source files are configure dependencies, so adding
    # or renaming a literal TEST_CASE automatically refreshes the list.
    function(mcpa_register_doctest_cases target prefix labels skip_return_code)
        foreach(source IN LISTS ARGN)
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
            file(STRINGS "${source}" case_lines
                REGEX "^[ \t]*TEST_CASE\\(\"[^\"]+\"\\)")
            foreach(case_line IN LISTS case_lines)
                string(REGEX REPLACE
                    "^[ \t]*TEST_CASE\\(\"([^\"]+)\"\\).*$" "\\1"
                    case_name "${case_line}")
                set(test_name "${prefix}: ${case_name}")
                add_test(NAME "${test_name}"
                    COMMAND ${target}
                        "--test-case=${case_name}"
                        --no-version
                        --no-intro)
                set_tests_properties("${test_name}" PROPERTIES LABELS "${labels}")
                if(NOT "${skip_return_code}" STREQUAL "")
                    set_tests_properties("${test_name}" PROPERTIES
                        SKIP_RETURN_CODE "${skip_return_code}")
                endif()
            endforeach()
        endforeach()
    endfunction()

    mcpa_register_doctest_cases(
        mcpa_cpu_tests "CPU" "cpu;fast" "" ${MCPA_CPU_TEST_SOURCES})
    mcpa_register_doctest_cases(
        mcpa_cuda_tests "GPU" "cuda;gpu" "77" ${MCPA_CUDA_TEST_SOURCES})
    mcpa_register_doctest_cases(
        mcpa_cpu_tests "CHECKPOINT" "checkpoint" ""
        ${MCPA_CPU_CHECKPOINT_SOURCE})
    mcpa_register_doctest_cases(
        mcpa_cuda_tests "CHECKPOINT" "checkpoint" "77"
        ${MCPA_CUDA_CHECKPOINT_SOURCE})

    add_test(
        NAME "CHECKPOINT: Cluster-style process interruption resumes identical outputs"
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_checkpoint_process.sh"
            $<TARGET_FILE:main_bw> $<TARGET_FILE:mcpa_cuda_tests>)
    set_tests_properties(
        "CHECKPOINT: Cluster-style process interruption resumes identical outputs"
        PROPERTIES LABELS "checkpoint;integration" SKIP_RETURN_CODE 77)
endif()

if(MCPA_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
