cmake_minimum_required(VERSION 3.22)

project(MCPA_2DIsing VERSION 0.1.0 LANGUAGES CXX CUDA)

include(CTest)
find_package(CUDAToolkit 12.4 REQUIRED)

option(MCPA_BUILD_TESTS "Build the 2D Ising CPU exact-oracle suite" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures" FORCE)
endif()

# This is the authoritative model target. The mislabelled Blume-Capel
# prototype formerly on this branch remains available in Git history only.
add_library(mcpa_ising2d STATIC
    src/ising2d_model.cu
    src/ising2d_cuda.cu
    src/ising2d_resampling.cpp
    src/ising2d_checkpoint.cpp
)
target_include_directories(mcpa_ising2d PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(mcpa_ising2d PUBLIC CUDA::cudart CUDA::curand)
target_compile_options(mcpa_ising2d PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)
set_target_properties(mcpa_ising2d PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(main_2d_ising main/main_2d_ising.cpp)
target_link_libraries(main_2d_ising PRIVATE mcpa_ising2d)
target_compile_options(main_2d_ising PRIVATE -Wall -Wextra -Wpedantic)

if(BUILD_TESTING AND MCPA_BUILD_TESTS)
    add_executable(mcpa_ising2d_cpu_tests
        tests/test_main.cpp
        tests/test_ising2d_model.cpp
        tests/test_ising2d_resampling.cpp
        tests/test_ising2d_checkpoint.cpp
    )
    target_link_libraries(mcpa_ising2d_cpu_tests PRIVATE mcpa_ising2d)
    target_compile_options(mcpa_ising2d_cpu_tests PRIVATE -Wall -Wextra -Wpedantic)

    set(MCPA_ISING2D_CPU_CASES
        "Exact L=3 DOS matches all 512 states"
        "Constructed L=3 and L=4 states have exact observables"
        "Local flip delta matches full energy recomputation"
        "Periodic square-torus geometry is exact"
        "Spectrum endpoints and strict sentinels are exact"
        "Spin domain and lattice size are enforced"
        "Resampling PCG state replays parents and families exactly"
        "Strict resampling distinguishes terminal and no-next-shell states"
        "Int64 resampling shells do not narrow or overflow"
    )

    foreach(case_name IN LISTS MCPA_ISING2D_CPU_CASES)
        add_test(
            NAME "CPU: ${case_name}"
            COMMAND mcpa_ising2d_cpu_tests "--test=${case_name}"
        )
        set_tests_properties("CPU: ${case_name}" PROPERTIES LABELS "cpu;exact;fast")
    endforeach()

    set(MCPA_ISING2D_CHECKPOINT_CPU_CASES
        "Checkpoint roundtrip preserves every Ising and RNG field"
        "Checkpoint rotation retains current and previous generations"
        "Checkpoint CRC corruption falls back to the previous generation"
        "Checkpoint loader recovers a synced completed temporary generation"
        "Checkpoint done marker is durable and idempotent"
        "Checkpoint semantic validation rejects physical corruption"
    )
    foreach(case_name IN LISTS MCPA_ISING2D_CHECKPOINT_CPU_CASES)
        add_test(
            NAME "CHECKPOINT CPU: ${case_name}"
            COMMAND mcpa_ising2d_cpu_tests "--test=${case_name}"
        )
        set_tests_properties("CHECKPOINT CPU: ${case_name}" PROPERTIES
            LABELS "checkpoint;cpu;fast"
        )
    endforeach()

    add_executable(mcpa_ising2d_cuda_tests
        tests/test_cuda_main.cpp
        tests/test_ising2d_cuda.cu
    )
    target_link_libraries(mcpa_ising2d_cuda_tests PRIVATE mcpa_ising2d)
    target_compile_options(mcpa_ising2d_cuda_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra;-Xptxas=-warn-spills>
    )

    set(MCPA_ISING2D_CUDA_CASES
        "CUDA Philox initialization is byte-exact"
        "CUDA spin initialization has exact domain energy flips and RNG"
        "CUDA full energy matches an independent reference"
        "CUDA replica copying uses an immutable source snapshot"
        "Checkpoint restores bitwise cooling and heating continuation"
        "Cooling open trajectory matches the independent reference"
        "Cooling frozen trajectory matches the independent reference"
        "Cooling mixed trajectory matches the independent reference"
        "Heating open trajectory matches the independent reference"
        "Heating frozen trajectory matches the independent reference"
        "Heating mixed trajectory matches the independent reference"
    )

    foreach(case_name IN LISTS MCPA_ISING2D_CUDA_CASES)
        add_test(
            NAME "GPU: ${case_name}"
            COMMAND mcpa_ising2d_cuda_tests "--test=${case_name}"
        )
        set_tests_properties("GPU: ${case_name}" PROPERTIES
            LABELS "cuda;gpu"
            SKIP_RETURN_CODE 77
        )
    endforeach()
    set_tests_properties(
        "GPU: Checkpoint restores bitwise cooling and heating continuation"
        PROPERTIES LABELS "cuda;gpu;checkpoint" SKIP_RETURN_CODE 77
    )

    add_executable(mcpa_ising2d_process_tests
        tests/test_process_main.cpp
        tests/test_ising2d_process.cpp
    )
    target_link_libraries(mcpa_ising2d_process_tests PRIVATE CUDA::cudart)
    target_compile_options(mcpa_ising2d_process_tests PRIVATE
        -Wall -Wextra -Wpedantic
    )
    add_dependencies(mcpa_ising2d_process_tests main_2d_ising)

    set(MCPA_ISING2D_PROCESS_CASES
        "Bounded L=3 cooling process writes one complete terminal dataset"
        "Bounded L=3 heating process writes one complete terminal dataset"
    )
    foreach(case_name IN LISTS MCPA_ISING2D_PROCESS_CASES)
        add_test(
            NAME "PROCESS: ${case_name}"
            COMMAND mcpa_ising2d_process_tests
                    "--binary=$<TARGET_FILE:main_2d_ising>"
                    "--test=${case_name}"
        )
        set_tests_properties("PROCESS: ${case_name}" PROPERTIES
            LABELS "cuda;gpu;process;integration"
            SKIP_RETURN_CODE 77
        )
    endforeach()

    set(MCPA_ISING2D_CHECKPOINT_PROCESS_CASES
        "SIGKILL cooling restart reproduces all outputs and done is a no-op"
        "SIGKILL heating restart reproduces all outputs and done is a no-op"
    )
    foreach(case_name IN LISTS MCPA_ISING2D_CHECKPOINT_PROCESS_CASES)
        add_test(
            NAME "CHECKPOINT PROCESS: ${case_name}"
            COMMAND mcpa_ising2d_process_tests
                    "--binary=$<TARGET_FILE:main_2d_ising>"
                    "--test=${case_name}"
        )
        set_tests_properties("CHECKPOINT PROCESS: ${case_name}" PROPERTIES
            LABELS "checkpoint;cuda;gpu;process;integration"
            SKIP_RETURN_CODE 77
            TIMEOUT 60
        )
    endforeach()
endif()
