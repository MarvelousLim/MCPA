cmake_minimum_required(VERSION 3.22)

project(MCPA_1D_Ising VERSION 0.1.0 LANGUAGES CXX CUDA)

include(CTest)
find_package(CUDAToolkit 12.4 REQUIRED)

option(MCPA_BUILD_TESTS "Build the exact 1D Ising CPU oracle suite" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures" FORCE)
endif()

function(mcpa_configure_1d_ising_main target)
    add_executable(${target}
        src/ising1d_simulation.cu
        src/checkpoint_1d.cpp
        src/ising1d_output.cpp
    )
    target_link_libraries(${target} PRIVATE CUDA::cudart)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra;-Xptxas=-warn-spills>
    )
endfunction()

mcpa_configure_1d_ising_main(main_1d_ising)
mcpa_configure_1d_ising_main(main_1d_ising_legacy_l_squared)
target_compile_definitions(main_1d_ising_legacy_l_squared PRIVATE
    MCPA_1D_ISING_LEGACY_L_SQUARED_CLI=1
)

add_library(mcpa_1d_ising_test_api STATIC
    src/ising1d_simulation.cu
    src/checkpoint_1d.cpp
    src/ising1d_output.cpp
)
target_compile_definitions(mcpa_1d_ising_test_api PRIVATE MCPA_1D_ISING_NO_MAIN=1)
target_include_directories(mcpa_1d_ising_test_api PUBLIC src)
target_link_libraries(mcpa_1d_ising_test_api PUBLIC CUDA::cudart)
target_compile_options(mcpa_1d_ising_test_api PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra;-Xptxas=-warn-spills>
)

if(BUILD_TESTING AND MCPA_BUILD_TESTS)
    add_executable(mcpa_1d_ising_cpu_tests
        tests/test_main.cpp
        tests/test_cpu_ring.cpp
        tests/test_cpu_checkpoint.cpp
        tests/test_cpu_output.cpp
    )
    target_link_libraries(mcpa_1d_ising_cpu_tests PRIVATE
        mcpa_1d_ising_test_api
    )
    target_compile_options(mcpa_1d_ising_cpu_tests PRIVATE -Wall -Wextra -Wpedantic)

    set(MCPA_1D_ISING_CPU_CASES
        "Even N=6 exhaustive DOS matches the analytic ring DOS"
        "Odd N=5 exhaustive DOS matches the analytic ring DOS"
        "Small-ring DOS totals 2^N and has two ground states"
        "Every periodic state has even walls and E=-N+2w"
        "Local flip delta matches a full energy recomputation"
        "Periodic ring neighbors wrap at every site"
        "Cooling selects the next strict occupied shell"
        "Resample reports the exact integer cooling shell"
        "Resample reports bounded unique terminal states"
        "PCG resampling state replays the exact parent map"
        "PCG parent selection uses only surviving replicas"
        "Checkpoint round trip preserves packed Ising and RNG state"
        "Checkpoint rotation retains the previous complete generation"
        "Checkpoint CRC rejects corruption without a valid fallback"
        "Complete checkpoint temporary file is recoverable"
        "Checkpoint done marker suppresses completed trajectory resume"
        "Checkpoint rejects inconsistent replica semantics"
        "Checkpoint identity includes the detailed output cap"
        "Ising shell measurements preserve pre-resampling families"
    )

    foreach(case_name IN LISTS MCPA_1D_ISING_CPU_CASES)
        add_test(
            NAME "CPU: ${case_name}"
            COMMAND mcpa_1d_ising_cpu_tests "--test=${case_name}"
        )
        if(case_name MATCHES "[Cc]heckpoint")
            set(test_labels "cpu;checkpoint;fast")
        else()
            set(test_labels "cpu;fast")
        endif()
        set_tests_properties("CPU: ${case_name}" PROPERTIES LABELS "${test_labels}")
    endforeach()

    add_executable(mcpa_1d_ising_cuda_tests
        tests/test_cuda_main.cpp
        tests/test_cuda_trajectory.cu
        tests/test_cuda_checkpoint.cu
    )
    target_link_libraries(mcpa_1d_ising_cuda_tests PRIVATE
        mcpa_1d_ising_test_api
    )
    target_compile_options(mcpa_1d_ising_cuda_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra;-Xptxas=-warn-spills>
    )

    set(MCPA_1D_ISING_CUDA_CASES
        "CUDA initialization stays in domain with exact tracked energy"
        "Even N=6 spins energies flips and Philox match the reference"
        "Odd N=5 spins energies flips and Philox match the reference"
        "CUDA checkpoint continuation is bitwise equal after a copied shell"
    )

    foreach(case_name IN LISTS MCPA_1D_ISING_CUDA_CASES)
        add_test(
            NAME "GPU: ${case_name}"
            COMMAND mcpa_1d_ising_cuda_tests "--test=${case_name}"
        )
        if(case_name MATCHES "checkpoint")
            set(cuda_test_labels "cuda;gpu;checkpoint")
        else()
            set(cuda_test_labels "cuda;gpu")
        endif()
        set_tests_properties("GPU: ${case_name}" PROPERTIES
            LABELS "${cuda_test_labels}"
            SKIP_RETURN_CODE 77
        )
    endforeach()

    add_test(
        NAME "GPU: Canonical and legacy CLIs emit exactly three versioned outputs"
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_process_cli.sh"
            $<TARGET_FILE:main_1d_ising>
            $<TARGET_FILE:main_1d_ising_legacy_l_squared>
            $<TARGET_FILE:mcpa_1d_ising_cuda_tests>
    )
    set_tests_properties(
        "GPU: Canonical and legacy CLIs emit exactly three versioned outputs"
        PROPERTIES LABELS "cuda;gpu;integration" SKIP_RETURN_CODE 77 TIMEOUT 45
    )

    add_test(
        NAME "GPU: Killed checkpoint process preserves deterministic three-file output"
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_checkpoint_process.sh"
            $<TARGET_FILE:main_1d_ising>
            $<TARGET_FILE:mcpa_1d_ising_cuda_tests>
    )
    set_tests_properties(
        "GPU: Killed checkpoint process preserves deterministic three-file output"
        PROPERTIES LABELS "cuda;gpu;checkpoint;integration"
                   SKIP_RETURN_CODE 77 TIMEOUT 45
    )
endif()
