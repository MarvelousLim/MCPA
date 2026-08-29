cmake_minimum_required(VERSION 3.22)

project(MCPA_Potts VERSION 0.1.0 LANGUAGES CXX CUDA)

include(CTest)
find_package(CUDAToolkit 12.4 REQUIRED)

option(MCPA_BUILD_TESTS "Build the Potts CPU and CUDA oracle suites" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures" FORCE)
endif()

add_library(mcpa_potts STATIC
    lib/potts_lib.cu
    lib/potts_checkpoint.cpp
    lib/potts_output.cpp
    lib/mem_estimate.cpp
)
target_include_directories(mcpa_potts PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/lib")
target_link_libraries(mcpa_potts PUBLIC CUDA::cudart)
target_compile_options(mcpa_potts PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra;-Xptxas=-warn-spills>
)
set_target_properties(mcpa_potts PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(main_potts main/main.cpp)
target_link_libraries(main_potts PRIVATE mcpa_potts)

if(BUILD_TESTING AND MCPA_BUILD_TESTS)
    add_executable(mcpa_potts_cpu_tests
        tests/test_main.cpp
        tests/test_cpu_potts.cpp
        tests/test_cpu_checkpoint.cpp
        tests/test_cpu_output.cpp
    )
    target_link_libraries(mcpa_potts_cpu_tests PRIVATE mcpa_potts)
    target_compile_options(mcpa_potts_cpu_tests PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(mcpa_potts_cuda_tests
        tests/test_cuda_main.cpp
        tests/test_cuda_trajectory.cu
        tests/test_cuda_checkpoint.cu
    )
    target_include_directories(mcpa_potts_cuda_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_link_libraries(mcpa_potts_cuda_tests PRIVATE mcpa_potts)
    target_compile_options(mcpa_potts_cuda_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )

    set(MCPA_POTTS_CPU_CASES
        "Exact L=2 DOS is correct for q=2,3,4"
        "Exact L=3 DOS is correct for q=2,3,4"
        "Byte spins preserve the q=2,3,4 domains"
        "Periodic square-lattice neighbors are exact"
        "Full energies match an independent bond oracle"
        "Local energy changes match an independent oracle"
        "q=2 Potts energy obeys the affine Ising map"
        "Cooling heating and terminal shell semantics are strict"
        "Resampling RNG state replays parent choices exactly"
        "Potts checkpoint round trip preserves byte spins and both RNG streams"
        "Potts checkpoint CRC corruption falls back to retained previous"
        "Potts checkpoint complete temporary generation is recoverable"
        "Potts checkpoint done marker suppresses completed restart"
        "Potts checkpoint semantic validation rejects CRC-valid bad spins"
        "Potts checkpoint semantic validation covers q energy and order"
        "Potts checkpoint identity includes the detailed cap"
        "Potts three-file schemas are rectangular and GPU metadata appears once"
        "Potts detailed cap is deterministic and preserves pre-resampling families"
        "Potts shell statistics select exactly the new energy subset"
        "q=2 Potts scalar order is exactly absolute Ising magnetization"
    )

    foreach(case_name IN LISTS MCPA_POTTS_CPU_CASES)
        add_test(
            NAME "CPU: ${case_name}"
            COMMAND mcpa_potts_cpu_tests "--test=${case_name}"
        )
        if(case_name MATCHES "checkpoint")
            set(cpu_test_labels "cpu;checkpoint;fast")
        else()
            set(cpu_test_labels "cpu;fast")
        endif()
        set_tests_properties("CPU: ${case_name}" PROPERTIES LABELS "${cpu_test_labels}")
    endforeach()

    set(MCPA_POTTS_CUDA_CASES
        "CUDA initialization and tracked energies match independent oracles"
        "CUDA cooling and heating open and frozen trajectories are bitwise exact"
        "CUDA mixed cooling and heating trajectories are bitwise exact"
        "CUDA Potts q=2,3,4 checkpoint continuations are bitwise exact"
    )

    foreach(case_name IN LISTS MCPA_POTTS_CUDA_CASES)
        add_test(
            NAME "CUDA: ${case_name}"
            COMMAND mcpa_potts_cuda_tests "--test=${case_name}"
        )
        if(case_name MATCHES "checkpoint")
            set(cuda_test_labels "cuda;gpu;checkpoint")
        else()
            set(cuda_test_labels "cuda;gpu")
        endif()
        set_tests_properties("CUDA: ${case_name}" PROPERTIES
            LABELS "${cuda_test_labels}"
            SKIP_RETURN_CODE 77
            TIMEOUT 45
        )
    endforeach()

    add_test(
        NAME "PROCESS: Invalid Potts parameters fail before CUDA setup"
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_process.sh"
            $<TARGET_FILE:main_potts> invalid-parameters
    )
    set_tests_properties(
        "PROCESS: Invalid Potts parameters fail before CUDA setup"
        PROPERTIES LABELS "process;fast" TIMEOUT 15)

    add_test(
        NAME "PROCESS: Potts output directories are created before CUDA setup"
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_process.sh"
            $<TARGET_FILE:main_potts> output-directories
    )
    set_tests_properties(
        "PROCESS: Potts output directories are created before CUDA setup"
        PROPERTIES LABELS "process;fast" TIMEOUT 15)

    add_test(
        NAME "PROCESS: Invalid Potts output root fails before CUDA setup"
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_process.sh"
            $<TARGET_FILE:main_potts> invalid-output-root
    )
    set_tests_properties(
        "PROCESS: Invalid Potts output root fails before CUDA setup"
        PROPERTIES LABELS "process;fast" TIMEOUT 15)

    foreach(direction IN ITEMS 0 1)
        if(direction EQUAL 0)
            set(direction_name "cooling")
        else()
            set(direction_name "heating")
        endif()
        add_test(
            NAME "PROCESS: Potts ${direction_name} three-file output contract"
            COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_output_process.sh"
                $<TARGET_FILE:main_potts>
                $<TARGET_FILE:mcpa_potts_cuda_tests> ${direction}
        )
        set_tests_properties(
            "PROCESS: Potts ${direction_name} three-file output contract"
            PROPERTIES LABELS "process;cuda;gpu;output;integration"
                       SKIP_RETURN_CODE 77 TIMEOUT 45)
        add_test(
            NAME "PROCESS: Killed Potts ${direction_name} checkpoint resumes byte-identical outputs"
            COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_checkpoint_process.sh"
                $<TARGET_FILE:main_potts>
                $<TARGET_FILE:mcpa_potts_cuda_tests> ${direction}
        )
        set_tests_properties(
            "PROCESS: Killed Potts ${direction_name} checkpoint resumes byte-identical outputs"
            PROPERTIES LABELS "process;cuda;gpu;checkpoint;integration"
                       SKIP_RETURN_CODE 77 TIMEOUT 60)
    endforeach()
endif()
