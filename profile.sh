#!/usr/bin/env bash
set -euo pipefail

profiler="${1:-ncu}"
preset="${2:-smoke}"
variant="${3:-candidate}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${MCPA_BUILD_DIR:-${repo_dir}/build}"
binary="${build_dir}/benchmarks/mcpa_equilibrate_bench"
report_dir="${MCPA_PROFILE_DIR:-${repo_dir}/optimization/artifacts}"
cuda_compiler="${CUDACXX:-/usr/local/cuda-12.4/bin/nvcc}"
host_compiler="${CXX:-/usr/bin/g++-11}"
cuda_architectures="${MCPA_CUDA_ARCHITECTURES:-70}"
mkdir -p "${report_dir}"

case "${variant}" in candidate|reference) ;;
    *) echo "Usage: ./profile.sh [ncu|nsys] [smoke|production|working-set] [candidate|reference]" >&2; exit 2 ;;
esac

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER="${cuda_compiler}" \
    -DCMAKE_CUDA_HOST_COMPILER="${host_compiler}" \
    -DCMAKE_CXX_COMPILER="${host_compiler}" \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
    -DMCPA_BUILD_TESTS=OFF \
    -DMCPA_BUILD_BENCHMARKS=ON >/dev/null
cmake --build "${build_dir}" --target mcpa_equilibrate_bench --parallel "${MCPA_BUILD_JOBS:-2}" >/dev/null

case "${profiler}" in
    ncu)
        command -v ncu >/dev/null || { echo "ERROR: ncu is not on PATH" >&2; exit 2; }
        report_base="${MCPA_PROFILE_BASE:-${report_dir}/ncu-${preset}-${variant}-$(date -u +%Y%m%dT%H%M%SZ)}"
        echo "Nsight Compute: look first at Duration, DRAM Throughput, Memory Throughput, and Achieved Occupancy."
        echo "A memory-bound kernel has high memory throughput while Compute (SM) remains low."
        exec ncu \
            --force-overwrite \
            --export "${report_base}" \
            --target-processes all \
            --kernel-name 'regex:.*equilibrate_kernel.*' \
            --launch-count 1 \
            --section SpeedOfLight \
            --section MemoryWorkloadAnalysis \
            --section Occupancy \
            "${binary}" --preset "${preset}" --variant "${variant}" \
            --warmups 1 --repeats 1 --skip-check
        ;;
    nsys)
        command -v nsys >/dev/null || { echo "ERROR: nsys is not on PATH" >&2; exit 2; }
        report_base="${MCPA_PROFILE_BASE:-${report_dir}/nsys-${preset}-${variant}-$(date -u +%Y%m%dT%H%M%SZ)}"
        echo "Nsight Systems: the summary below separates kernel time from CUDA API/synchronization time."
        nsys profile --force-overwrite=true --sample=none --trace=cuda \
            --output "${report_base}" \
            "${binary}" --preset "${preset}" --variant "${variant}" \
            --warmups 1 --repeats 1 --skip-check
        nsys stats --report cuda_gpu_kern_sum,cuda_api_sum "${report_base}.nsys-rep"
        echo "Raw report kept at ${report_base}.nsys-rep"
        ;;
    *)
        echo "Usage: ./profile.sh [ncu|nsys] [smoke|production|working-set] [candidate|reference]" >&2
        exit 2
        ;;
esac
