#!/usr/bin/env bash
# Build the Baxter--Wu executable from this self-contained MCPA checkout.
# This script never submits a job.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${MCPA_BUILD_DIR:-${REPO_DIR}/build/cluster}"
OUTPUT_EXE="${MCPA_OUTPUT_EXE:-${REPO_DIR}/build/2DBaxterWu.exe}"

if command -v module >/dev/null 2>&1; then
    # The HSE default gnu8 module injects an old libstdc++ that cannot start
    # the current CMake module. Use a CUDA-12.4-compatible compiler module.
    if module is-loaded gnu8 >/dev/null 2>&1 &&
       module is-avail gnu12/12.1 >/dev/null 2>&1; then
        echo "Switching cluster compiler module: gnu8 -> gnu12/12.1"
        module swap gnu8 gnu12/12.1
    fi
    echo "Loading cluster toolchain module: CUDA/12.4"
    module load CUDA/12.4
    if ! command -v cmake >/dev/null 2>&1; then
        echo "Loading cluster build module: cmake/3.31.8"
        module load cmake/3.31.8
    fi
fi

find_supported_gcc() {
    local candidate path version major
    for candidate in "$@"; do
        path="$(command -v "${candidate}" 2>/dev/null || true)"
        [[ -n "${path}" ]] || continue
        version="$(${path} -dumpfullversion -dumpversion 2>/dev/null || true)"
        major="${version%%.*}"
        if [[ "${major}" =~ ^[0-9]+$ ]] && (( major >= 6 && major <= 13 )); then
            printf '%s\n' "${path}"
            return 0
        fi
    done
    return 1
}

if [[ -x /usr/local/cuda-12.4/bin/nvcc ]]; then
    NVCC=/usr/local/cuda-12.4/bin/nvcc
else
    NVCC="$(command -v nvcc || true)"
fi
if [[ -n "${MCPA_HOST_CC:-}" ]]; then
    HOST_CC="$(command -v "${MCPA_HOST_CC}" 2>/dev/null || true)"
else
    HOST_CC="$(find_supported_gcc gcc-11 gcc || true)"
fi
if [[ -n "${MCPA_HOST_CXX:-}" ]]; then
    HOST_CXX="$(command -v "${MCPA_HOST_CXX}" 2>/dev/null || true)"
else
    HOST_CXX="$(find_supported_gcc g++-11 g++ || true)"
fi

[[ -n "${NVCC}" ]] || { echo "ERROR: nvcc was not found" >&2; exit 1; }
command -v cmake >/dev/null 2>&1 || {
    echo "ERROR: CMake was not found; load a CMake module version 3.22 or newer." >&2
    exit 1
}
CMAKE_VERSION_LINE=""
if ! CMAKE_VERSION_LINE="$(cmake --version 2>&1 | head -n 1)"; then
    echo "ERROR: CMake exists but cannot start with the loaded compiler libraries." >&2
    echo "CMake reported: ${CMAKE_VERSION_LINE}" >&2
    command -v module >/dev/null 2>&1 && module list >&2 || true
    exit 1
fi
if [[ -z "${HOST_CC}" || -z "${HOST_CXX}" ]]; then
    echo "ERROR: a CUDA-12.4-compatible GCC toolchain was not found." >&2
    echo "Supported GCC majors are 6 through 13; check 'module avail gnu'." >&2
    echo "You may also set MCPA_HOST_CC and MCPA_HOST_CXX explicitly." >&2
    exit 1
fi

if ! "${NVCC}" --version | grep -q 'release 12\.4'; then
    echo "ERROR: CUDA 12.4 is required; found:" >&2
    "${NVCC}" --version >&2
    exit 1
fi
C_VERSION="$("${HOST_CC}" -dumpfullversion -dumpversion)"
CXX_VERSION="$("${HOST_CXX}" -dumpfullversion -dumpversion)"
C_MAJOR="${C_VERSION%%.*}"
CXX_MAJOR="${CXX_VERSION%%.*}"
if [[ ! "${C_MAJOR}" =~ ^[0-9]+$ ]] || [[ ! "${CXX_MAJOR}" =~ ^[0-9]+$ ]] ||
   (( C_MAJOR < 6 || C_MAJOR > 13 || CXX_MAJOR < 6 || CXX_MAJOR > 13 )); then
    echo "ERROR: CUDA 12.4 supports GCC 6.x through 13.2; found C=${C_VERSION}, C++=${CXX_VERSION}" >&2
    exit 1
fi
if (( C_MAJOR != CXX_MAJOR )); then
    echo "ERROR: C and C++ compiler major versions differ: C=${C_VERSION}, C++=${CXX_VERSION}" >&2
    exit 1
fi

echo "MCPA source: ${REPO_DIR}"
echo "Build dir:   ${BUILD_DIR}"
echo "CUDA:       $("${NVCC}" --version | grep release)"
echo "CMake:      ${CMAKE_VERSION_LINE}"
if (( CXX_MAJOR != 11 )); then
    echo "Note: using cluster GCC ${CXX_VERSION}; GCC 11 remains the local reference."
fi
echo "Host C++:   $("${HOST_CXX}" --version | head -n 1)"

mkdir -p "$(dirname -- "${OUTPUT_EXE}")"

cmake -S "${REPO_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${HOST_CC}" \
    -DCMAKE_CXX_COMPILER="${HOST_CXX}" \
    -DCMAKE_CUDA_COMPILER="${NVCC}" \
    -DCMAKE_CUDA_HOST_COMPILER="${HOST_CXX}" \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;89;90" \
    -DBUILD_TESTING=OFF \
    -DMCPA_BUILD_TESTS=OFF \
    -DMCPA_BUILD_BENCHMARKS=OFF
cmake --build "${BUILD_DIR}" --target main_bw --parallel "${MCPA_BUILD_JOBS:-4}"
cmake -E copy "${BUILD_DIR}/main_bw" "${OUTPUT_EXE}"

echo "PASS: built ${OUTPUT_EXE}"
