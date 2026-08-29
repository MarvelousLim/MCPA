#!/usr/bin/env bash
set -euo pipefail

mode="${1:-cpu}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${MCPA_BUILD_DIR:-${repo_dir}/build}"
cuda_compiler="${CUDACXX:-/usr/local/cuda-12.4/bin/nvcc}"
host_compiler="${CXX:-/usr/bin/g++-11}"
cuda_architectures="${MCPA_CUDA_ARCHITECTURES:-70}"

case "${mode}" in
    cpu|gpu|checkpoint|all|list) ;;
    *)
        echo "Usage: ./test.sh [cpu|gpu|checkpoint|all|list]" >&2
        exit 2
        ;;
esac

color_mode="${MCPA_COLOR:-auto}"
case "${color_mode}" in
    auto|always|never) ;;
    *) echo "ERROR: MCPA_COLOR must be auto, always, or never" >&2; exit 2 ;;
esac
use_color=0
if [[ "${color_mode}" == "always" ]] ||
   [[ "${color_mode}" == "auto" && -t 1 && "${TERM:-dumb}" != "dumb" && -z "${NO_COLOR:-}" ]]; then
    use_color=1
fi
if (( use_color )); then
    green=$'\033[32m'; red=$'\033[31m'; yellow=$'\033[33m'; reset=$'\033[0m'
else
    green=""; red=""; yellow=""; reset=""
fi

print_verdict() {
    local verdict="$1" color="$2" name="$3"
    printf '%s%s%s: %s\n' "${color}" "${verdict}" "${reset}" "${name}"
}

if [[ ! -x "${cuda_compiler}" ]]; then
    echo "ERROR: CUDA compiler not found: ${cuda_compiler}" >&2
    exit 2
fi
if [[ ! -x "${host_compiler}" ]]; then
    echo "ERROR: host C++ compiler not found: ${host_compiler}" >&2
    exit 2
fi

command_log="$(mktemp "${TMPDIR:-/tmp}/mcpa-1d-test.XXXXXX")"
trap 'rm -f -- "${command_log}"' EXIT

if ! cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER="${cuda_compiler}" \
    -DCMAKE_CUDA_HOST_COMPILER="${host_compiler}" \
    -DCMAKE_CXX_COMPILER="${host_compiler}" \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
    -DMCPA_BUILD_TESTS=ON >"${command_log}" 2>&1; then
    echo "FAIL: CMake configuration" >&2
    cat "${command_log}" >&2
    exit 1
fi

case "${mode}" in
    cpu) targets=(mcpa_1d_ising_cpu_tests main_1d_ising main_1d_ising_legacy_l_squared) ;;
    gpu) targets=(mcpa_1d_ising_cuda_tests main_1d_ising main_1d_ising_legacy_l_squared) ;;
    checkpoint|all|list) targets=(mcpa_1d_ising_cpu_tests mcpa_1d_ising_cuda_tests main_1d_ising main_1d_ising_legacy_l_squared) ;;
esac

if ! cmake --build "${build_dir}" --target "${targets[@]}" \
    --parallel "${MCPA_BUILD_JOBS:-2}" >"${command_log}" 2>&1; then
    echo "FAIL: build" >&2
    cat "${command_log}" >&2
    exit 1
fi

run_ctest_compact() {
    local list_log test_log status test_number test_name overall=0 found=0
    list_log="$(mktemp "${TMPDIR:-/tmp}/mcpa-1d-ctest-list.XXXXXX")"
    test_log="$(mktemp "${TMPDIR:-/tmp}/mcpa-1d-ctest-case.XXXXXX")"
    if ! ctest "$@" -N >"${list_log}" 2>&1; then
        print_verdict "FAILED" "${red}" "CTest discovery"
        cat "${list_log}" >&2
        rm -f -- "${list_log}" "${test_log}"
        return 1
    fi
    while IFS=$'\t' read -r test_number test_name; do
        found=1
        if ctest "$@" -I "${test_number},${test_number}" >"${test_log}" 2>&1; then
            status=0
        else
            status=$?
        fi
        if grep -Eq '\*\*\*Skipped|[[:space:]]Skipped[[:space:]]' "${test_log}"; then
            print_verdict "SKIPPED" "${yellow}" "${test_name}"
        elif (( status == 0 )); then
            print_verdict "PASSED" "${green}" "${test_name}"
        else
            print_verdict "FAILED" "${red}" "${test_name}"
            printf '%s\n' "=== FAILURE DETAILS: ${test_name} ===" >&2
            cat "${test_log}" >&2
            overall=1
        fi
    done < <(sed -n -E 's/^[[:space:]]*Test +#([0-9]+):[[:space:]]*(.*)$/\1\t\2/p' "${list_log}")
    if (( found == 0 )); then
        print_verdict "FAILED" "${red}" "CTest discovery returned no matching contracts"
        cat "${list_log}" >&2
        overall=1
    fi
    rm -f -- "${list_log}" "${test_log}"
    return "${overall}"
}

case "${mode}" in
    list)
        ctest --test-dir "${build_dir}" --show-only
        ;;
    cpu)
        run_ctest_compact --test-dir "${build_dir}" -L cpu \
            --output-on-failure --timeout "${MCPA_TEST_TIMEOUT:-45}"
        ;;
    gpu)
        run_ctest_compact --test-dir "${build_dir}" -L gpu \
            --output-on-failure --timeout "${MCPA_TEST_TIMEOUT:-45}"
        ;;
    checkpoint)
        run_ctest_compact --test-dir "${build_dir}" -L checkpoint \
            --output-on-failure --timeout "${MCPA_TEST_TIMEOUT:-45}"
        ;;
    all)
        run_ctest_compact --test-dir "${build_dir}" \
            --output-on-failure --timeout "${MCPA_TEST_TIMEOUT:-45}"
        ;;
esac
