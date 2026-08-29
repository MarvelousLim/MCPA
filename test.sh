#!/usr/bin/env bash
set -euo pipefail

mode="${1:-cpu}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${MCPA_BUILD_DIR:-${repo_dir}/build}"
cuda_compiler="${CUDACXX:-/usr/local/cuda-12.4/bin/nvcc}"
host_compiler="${CXX:-/usr/bin/g++-11}"
cuda_architectures="${MCPA_CUDA_ARCHITECTURES:-70}"
build_log="$(mktemp /tmp/mcpa-bc-build-XXXXXX.log)"
trap 'rm -f -- "${build_log}"' EXIT

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

run_quiet_build_step() {
    local label="$1"
    shift
    local status=0
    if "$@" >>"${build_log}" 2>&1; then
        return 0
    else
        status=$?
    fi
    cat "${build_log}" >&2
    print_verdict "FAILED" "${red}" "${label}" >&2
    exit "${status}"
}

run_named_suite() {
    local binary="$1"
    local test_name output status pass_line
    while IFS= read -r test_name; do
        set +e
        output="$("${binary}" "--test=${test_name}" 2>&1)"
        status=$?
        set -e
        if [[ ${status} -eq 0 ]]; then
            pass_line="$(printf '%s\n' "${output}" | grep '^PASS:' | tail -n 1 || true)"
            if [[ -z "${pass_line}" ]]; then
                print_verdict "FAILED" "${red}" "${test_name} produced no PASS line"
                printf '%s\n' "${output}" >&2
                exit 1
            fi
            print_verdict "PASSED" "${green}" "${test_name}"
        elif [[ ${status} -eq 77 ]]; then
            print_verdict "SKIPPED" "${yellow}" "${test_name} (no accessible NVIDIA GPU)"
        else
            print_verdict "FAILED" "${red}" "${test_name}"
            printf '%s\n' "${output}" >&2
            exit "${status}"
        fi
    done < <("${binary}" --list)
}

run_process_gate() {
    local output status
    set +e
    output="$("${repo_dir}/tests/test_process_checkpoint.sh" \
        "${build_dir}/main_bc" "${build_dir}/mcpa_bc_cuda_tests" 2>&1)"
    status=$?
    set -e
    if [[ ${status} -eq 0 ]]; then
        print_verdict "PASSED" "${green}" "Killed restart preserves all non-timing output"
        return 0
    fi
    if [[ ${status} -eq 77 ]]; then
        print_verdict "SKIPPED" "${yellow}" "Killed restart process gate (no accessible NVIDIA GPU)"
        return 0
    fi
    print_verdict "FAILED" "${red}" "Killed restart preserves all non-timing output"
    printf '%s\n' "${output}" >&2
    exit "${status}"
}

case "${mode}" in
    cpu|gpu|checkpoint|all|list) ;;
    *)
        echo "Usage: ./test.sh [cpu|gpu|checkpoint|all|list]" >&2
        exit 2
        ;;
esac

if [[ ! -x "${cuda_compiler}" ]]; then
    echo "ERROR: CUDA compiler not found: ${cuda_compiler}" >&2
    exit 2
fi
if [[ ! -x "${host_compiler}" ]]; then
    echo "ERROR: host C++ compiler not found: ${host_compiler}" >&2
    exit 2
fi

run_quiet_build_step "CMake configure" cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER="${cuda_compiler}" \
    -DCMAKE_CUDA_HOST_COMPILER="${host_compiler}" \
    -DCMAKE_CXX_COMPILER="${host_compiler}" \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
    -DMCPA_BUILD_TESTS=ON

run_quiet_build_step "CMake build" cmake --build "${build_dir}" --target mcpa_bc_cpu_tests mcpa_bc_cuda_tests \
    mcpa_bc_checkpoint_tests mcpa_bc_cuda_checkpoint_tests main_bc \
    --parallel "${MCPA_BUILD_JOBS:-2}"

cpu_binary="${build_dir}/mcpa_bc_cpu_tests"
gpu_binary="${build_dir}/mcpa_bc_cuda_tests"
checkpoint_binary="${build_dir}/mcpa_bc_checkpoint_tests"
cuda_checkpoint_binary="${build_dir}/mcpa_bc_cuda_checkpoint_tests"
if [[ "${mode}" == "list" ]]; then
    "${cpu_binary}" --list
    "${gpu_binary}" --list || [[ $? -eq 77 ]]
    "${checkpoint_binary}" --list
    "${cuda_checkpoint_binary}" --list || [[ $? -eq 77 ]]
elif [[ "${mode}" == "cpu" ]]; then
    run_named_suite "${cpu_binary}"
elif [[ "${mode}" == "gpu" ]]; then
    run_named_suite "${gpu_binary}"
elif [[ "${mode}" == "checkpoint" ]]; then
    run_named_suite "${checkpoint_binary}"
    run_named_suite "${cuda_checkpoint_binary}"
    run_process_gate
else
    run_named_suite "${cpu_binary}"
    run_named_suite "${gpu_binary}"
    run_named_suite "${checkpoint_binary}"
    run_named_suite "${cuda_checkpoint_binary}"
    run_process_gate
fi
