cmake_minimum_required(VERSION 3.22)

project(MCPA_BlumeCapel VERSION 0.1.0 LANGUAGES CXX CUDA)

include(CTest)
find_package(CUDAToolkit 12.4 REQUIRED)

option(MCPA_BUILD_TESTS "Build the Blume-Capel correctness suites" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures" FORCE)
endif()

add_library(mcpa_blumecapel STATIC
    lib/blumeCapel_lib.cu
    lib/checkpoint.cpp
)
target_include_directories(mcpa_blumecapel PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/lib")
target_link_libraries(mcpa_blumecapel PUBLIC CUDA::cudart)
target_compile_options(mcpa_blumecapel PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra;-Xptxas=-warn-spills>
)
set_target_properties(mcpa_blumecapel PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(main_bc main/main_bc.cpp)
target_link_libraries(main_bc PRIVATE mcpa_blumecapel)

if(BUILD_TESTING AND MCPA_BUILD_TESTS)
    add_executable(mcpa_bc_cpu_tests
        tests/test_main.cpp
        tests/test_cpu_model.cpp
    )
    target_link_libraries(mcpa_bc_cpu_tests PRIVATE mcpa_blumecapel)
    target_compile_options(mcpa_bc_cpu_tests PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(mcpa_bc_cuda_tests
        tests/test_cuda_main.cpp
        tests/test_cuda_model.cu
    )

    add_executable(mcpa_bc_checkpoint_tests
        tests/test_main.cpp
        tests/test_cpu_checkpoint.cpp
    )
    add_executable(mcpa_bc_cuda_checkpoint_tests
        tests/test_cuda_main.cpp
        tests/test_cuda_checkpoint.cu
    )
    target_link_libraries(mcpa_bc_checkpoint_tests PRIVATE mcpa_blumecapel)
    target_compile_options(mcpa_bc_checkpoint_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(mcpa_bc_cuda_checkpoint_tests PRIVATE mcpa_blumecapel)
    target_compile_options(mcpa_bc_cuda_checkpoint_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )
    target_link_libraries(mcpa_bc_cuda_tests PRIVATE mcpa_blumecapel)
    target_compile_options(mcpa_bc_cuda_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )

    set(MCPA_BC_CPU_CASES
        "Exact L=3 joint DOS contains all 19683 states"
        "Exact L=3 joint DOS obeys spin-flip symmetry"
        "Fixed occupancy counts follow choose(9,k) times 2^k"
        "Constructed configurations have exact energy parts"
        "Rational D parsing preserves ordering and ties"
        "Spin proposals stay in the three-state domain"
        "Local energy changes match the independent oracle"
        "MCPA shell selection is strict in both directions"
        "Resampling RNG state restores parent choices exactly"
        "All-boundary populations return full culling without out-of-bounds access"
        "Aggregate statistics remain exact beyond 32-bit sums"
        "Output keeps legacy columns and appends exact shell identity"
        "Detailed cap modes preserve pre-resampling family evidence"
        "Periodic square-lattice neighbors are exact"
    )

    foreach(case_name IN LISTS MCPA_BC_CPU_CASES)
        add_test(
            NAME "CPU: ${case_name}"
            COMMAND mcpa_bc_cpu_tests "--test=${case_name}"
        )
        set_tests_properties("CPU: ${case_name}" PROPERTIES LABELS "cpu;fast")
    endforeach()

    set(MCPA_BC_CUDA_CASES
        "Initialized GPU population has valid spins and exact energy parts"
        "Cooling open trajectory and Philox state match the reference"
        "Cooling frozen trajectory and Philox state match the reference"
        "Heating open trajectory and Philox state match the reference"
        "Heating frozen trajectory and Philox state match the reference"
    )
    foreach(case_name IN LISTS MCPA_BC_CUDA_CASES)
        add_test(
            NAME "GPU: ${case_name}"
            COMMAND mcpa_bc_cuda_tests "--test=${case_name}"
        )
        set_tests_properties("GPU: ${case_name}" PROPERTIES
            LABELS "cuda;gpu" SKIP_RETURN_CODE 77)
    endforeach()

    set(MCPA_BC_CHECKPOINT_CASES
        "BC checkpoint roundtrip preserves complete trajectory state"
        "BC checkpoint rejects parameter mismatch"
        "BC checkpoint rejects CRC-valid impossible state without partial load"
        "BC checkpoint corruption falls back to previous generation"
        "BC checkpoint retains previous generation"
        "BC checkpoint recovers a valid temporary generation"
        "BC checkpoint done marker suppresses resume"
    )
    foreach(case_name IN LISTS MCPA_BC_CHECKPOINT_CASES)
        add_test(
            NAME "CHECKPOINT: ${case_name}"
            COMMAND mcpa_bc_checkpoint_tests "--test=${case_name}"
        )
        set_tests_properties("CHECKPOINT: ${case_name}" PROPERTIES
            LABELS "checkpoint;cpu" TIMEOUT 20)
    endforeach()

    set(MCPA_BC_CUDA_CHECKPOINT_CASES
        "Cooling checkpoint continuation is bitwise identical"
        "Heating checkpoint continuation is bitwise identical"
    )
    foreach(case_name IN LISTS MCPA_BC_CUDA_CHECKPOINT_CASES)
        add_test(
            NAME "CHECKPOINT GPU: ${case_name}"
            COMMAND mcpa_bc_cuda_checkpoint_tests "--test=${case_name}"
        )
        set_tests_properties("CHECKPOINT GPU: ${case_name}" PROPERTIES
            LABELS "checkpoint;cuda;gpu" SKIP_RETURN_CODE 77 TIMEOUT 45)
    endforeach()

    add_test(
        NAME "CHECKPOINT PROCESS: killed resume preserves all non-timing output"
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_process_checkpoint.sh"
                "$<TARGET_FILE:main_bc>" "$<TARGET_FILE:mcpa_bc_cuda_tests>"
    )
    set_tests_properties(
        "CHECKPOINT PROCESS: killed resume preserves all non-timing output"
        PROPERTIES LABELS "checkpoint;process;cuda;gpu" SKIP_RETURN_CODE 77 TIMEOUT 45)
endif()
