add_executable(mcpa_equilibrate_bench equilibrate_benchmark.cu)
target_link_libraries(mcpa_equilibrate_bench PRIVATE mcpa_baxterwu)
target_compile_options(mcpa_equilibrate_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-warn-spills>
)
