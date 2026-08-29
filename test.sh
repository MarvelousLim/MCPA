#!/usr/bin/env bash
set -euo pipefail

mode="${1:-all}"
if (($# > 0)); then shift; fi
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${MCPA_BUILD_DIR:-${repo_dir}/build}"
cuda_compiler="${CUDACXX:-/usr/local/cuda-12.4/bin/nvcc}"
host_compiler="${CXX:-/usr/bin/g++-11}"
cuda_architectures="${MCPA_CUDA_ARCHITECTURES:-70}"

case "${mode}" in
    all|correctness|cpu|gpu|checkpoint|benchmark|list) ;;
    *)
        echo "Usage: ./test.sh [all|correctness|cpu|gpu|checkpoint|benchmark|list] [benchmark preset/options]" >&2
        exit 2
        ;;
esac

color_mode="${MCPA_COLOR:-auto}"
case "${color_mode}" in
    auto|always|never) ;;
    *)
        echo "ERROR: MCPA_COLOR must be auto, always, or never" >&2
        exit 2
        ;;
esac

use_color=0
if [[ "${color_mode}" == "always" ]] ||
   [[ "${color_mode}" == "auto" && -t 1 && "${TERM:-dumb}" != "dumb" && -z "${NO_COLOR:-}" ]]; then
    use_color=1
fi
if (( use_color )); then
    green=$'\033[32m'
    red=$'\033[31m'
    yellow=$'\033[33m'
    reset=$'\033[0m'
else
    green=""
    red=""
    yellow=""
    reset=""
fi

print_verdict() {
    local verdict="$1" color="$2" name="$3"
    # Bash printf writes directly to stdout. Running one contract per process
    # means this line appears as soon as that contract exits, even through tee.
    printf '%s%s%s: %s\n' "${color}" "${verdict}" "${reset}" "${name}"
}

build_tests=ON
build_benchmarks=OFF
if [[ "${mode}" == "benchmark" ]]; then
    build_tests=OFF
    build_benchmarks=ON
elif [[ "${mode}" == "all" ]]; then
    build_benchmarks=ON
fi

if [[ ! -x "${cuda_compiler}" ]]; then
    echo "ERROR: CUDA compiler not found: ${cuda_compiler}" >&2
    exit 2
fi
if [[ ! -x "${host_compiler}" ]]; then
    echo "ERROR: host C++ compiler not found: ${host_compiler}" >&2
    exit 2
fi

command_log="$(mktemp "${TMPDIR:-/tmp}/mcpa-test-command.XXXXXX")"
trap 'rm -f -- "${command_log}"' EXIT

if ! cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER="${cuda_compiler}" \
    -DCMAKE_CUDA_HOST_COMPILER="${host_compiler}" \
    -DCMAKE_CXX_COMPILER="${host_compiler}" \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
    -DMCPA_BUILD_TESTS="${build_tests}" \
    -DMCPA_BUILD_BENCHMARKS="${build_benchmarks}" \
    >"${command_log}" 2>&1; then
    echo "ERROR: CMake configuration failed" >&2
    cat "${command_log}" >&2
    exit 1
fi

echo "Toolchain: $(${cuda_compiler} --version | sed -n '4p'); $(${host_compiler} --version | head -n 1)"

case "${mode}" in
    cpu) targets=(mcpa_cpu_tests) ;;
    gpu) targets=(mcpa_cuda_tests main_bw) ;;
    checkpoint) targets=(mcpa_cpu_tests mcpa_cuda_tests main_bw) ;;
    correctness|list) targets=(mcpa_cpu_tests mcpa_cuda_tests main_bw) ;;
    benchmark) targets=(mcpa_equilibrate_bench) ;;
    all) targets=(mcpa_cpu_tests mcpa_cuda_tests main_bw mcpa_equilibrate_bench) ;;
esac
if ! cmake --build "${build_dir}" --target "${targets[@]}" \
    --parallel "${MCPA_BUILD_JOBS:-2}" >"${command_log}" 2>&1; then
    echo "ERROR: build failed" >&2
    cat "${command_log}" >&2
    exit 1
fi

run_ctest_compact() {
    local list_log test_log status test_number test_name overall=0 found=0
    list_log="$(mktemp "${TMPDIR:-/tmp}/mcpa-ctest-list.XXXXXX")"
    test_log="$(mktemp "${TMPDIR:-/tmp}/mcpa-ctest-case.XXXXXX")"

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

run_benchmark() {
    local preset="${1:-smoke}"
    if (($# > 0)); then shift; fi
    case "${preset}" in
        smoke|production|working-set) ;;
        all)
            run_benchmark smoke "$@"
            run_benchmark production "$@"
            run_benchmark working-set "$@"
            return
            ;;
        *)
            echo "ERROR: benchmark preset must be smoke, production, working-set, or all" >&2
            return 2
            ;;
    esac
    "${build_dir}/benchmarks/mcpa_equilibrate_bench" --preset "${preset}" "$@"
}

run_optional_benchmark() {
    local status=0 benchmark_log direction="cooling"
    benchmark_log="$(mktemp "${TMPDIR:-/tmp}/mcpa-benchmark.XXXXXX")"
    if [[ " $* " == *" --heat "* ]]; then direction="heating"; fi

    if run_benchmark "$@" >"${benchmark_log}" 2>&1; then
        print_verdict "PASSED" "${green}" "Performance smoke (${direction}; informational timing)"
        rm -f -- "${benchmark_log}"
        return 0
    else
        status=$?
    fi
    if [[ "${status}" == "77" ]]; then
        print_verdict "SKIPPED" "${yellow}" "Performance smoke (${direction}; NVIDIA GPU unavailable)"
        rm -f -- "${benchmark_log}"
        return 0
    fi
    print_verdict "FAILED" "${red}" "Performance smoke (${direction})"
    cat "${benchmark_log}" >&2
    rm -f -- "${benchmark_log}"
    return "${status}"
}

case "${mode}" in
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
    list)
        ctest --test-dir "${build_dir}" --show-only
        ;;
    correctness)
        run_ctest_compact --test-dir "${build_dir}" \
            --output-on-failure --timeout "${MCPA_TEST_TIMEOUT:-45}"
        ;;
    benchmark)
        echo
        echo "=== PERFORMANCE (informational; no speed threshold) ==="
        if (($# > 0)); then
            run_benchmark "$@"
        else
            run_benchmark smoke
        fi
        ;;
    all)
        echo
        echo "=== CORRECTNESS CONTRACTS ==="
        run_ctest_compact --test-dir "${build_dir}" \
            --output-on-failure --timeout "${MCPA_TEST_TIMEOUT:-45}"
        echo
        echo "=== PERFORMANCE SMOKE CHECK (informational; no speed threshold) ==="
        run_optional_benchmark smoke --variant candidate --regime mixed
        run_optional_benchmark smoke --variant candidate --regime mixed --heat
        ;;
esac
