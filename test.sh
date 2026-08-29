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

log_file="$(mktemp "${TMPDIR:-/tmp}/mcpa-potts-test.XXXXXX")"
trap 'rm -f -- "${log_file}"' EXIT

if ! cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER="${cuda_compiler}" \
    -DCMAKE_CUDA_HOST_COMPILER="${host_compiler}" \
    -DCMAKE_CXX_COMPILER="${host_compiler}" \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
    -DMCPA_BUILD_TESTS=ON >"${log_file}" 2>&1; then
    print_verdict "FAILED" "${red}" "Configure Potts test build"
    cat "${log_file}" >&2
    exit 1
fi

targets=()
if [[ "${mode}" == "cpu" ]]; then
    targets=(mcpa_potts_cpu_tests main_potts)
elif [[ "${mode}" == "gpu" ]]; then
    targets=(mcpa_potts_cuda_tests main_potts)
else
    targets=(mcpa_potts_cpu_tests mcpa_potts_cuda_tests main_potts)
fi
if ! cmake --build "${build_dir}" --target "${targets[@]}" \
        --parallel "${MCPA_BUILD_JOBS:-2}" >"${log_file}" 2>&1; then
    print_verdict "FAILED" "${red}" "Build Potts test targets"
    cat "${log_file}" >&2
    exit 1
fi

if [[ "${mode}" == "list" ]]; then
    echo "CPU contracts:"
    "${build_dir}/mcpa_potts_cpu_tests" --list
    echo "CUDA contracts:"
    "${build_dir}/mcpa_potts_cuda_tests" --list
    echo "Process contracts:"
    printf '%s\n' \
        "Invalid Potts parameters fail before CUDA setup" \
        "Potts output directories are created before CUDA setup" \
        "Invalid Potts output root fails before CUDA setup" \
        "Potts cooling three-file output contract" \
        "Potts heating three-file output contract" \
        "Killed Potts cooling checkpoint resumes byte-identical outputs" \
        "Killed Potts heating checkpoint resumes byte-identical outputs"
    exit 0
fi

run_case() {
    local name="$1"
    shift
    set +e
    "$@" >"${log_file}" 2>&1
    local status=$?
    set -e
    if (( status == 0 )); then
        print_verdict "PASSED" "${green}" "${name}"
    elif (( status == 77 )); then
        print_verdict "SKIPPED" "${yellow}" "${name}"
    else
        print_verdict "FAILED" "${red}" "${name}"
        cat "${log_file}" >&2
        exit "${status}"
    fi
}

run_suite() {
    local binary="$1"
    local selection="${2:-}"
    while IFS= read -r name; do
        [[ -z "${selection}" || "${name}" == *"${selection}"* ]] || continue
        run_case "${name}" "${binary}" "--test=${name}"
    done < <("${binary}" --list)
}

if [[ "${mode}" == "cpu" || "${mode}" == "all" || "${mode}" == "checkpoint" ]]; then
    if [[ "${mode}" == "checkpoint" ]]; then
        run_suite "${build_dir}/mcpa_potts_cpu_tests" "checkpoint"
    else
        run_suite "${build_dir}/mcpa_potts_cpu_tests"
    fi
fi
if [[ "${mode}" == "cpu" || "${mode}" == "all" ]]; then
    run_case "Invalid Potts parameters fail before CUDA setup" \
        bash "${repo_dir}/tests/test_process.sh" \
        "${build_dir}/main_potts" invalid-parameters
    run_case "Potts output directories are created before CUDA setup" \
        bash "${repo_dir}/tests/test_process.sh" \
        "${build_dir}/main_potts" output-directories
    run_case "Invalid Potts output root fails before CUDA setup" \
        bash "${repo_dir}/tests/test_process.sh" \
        "${build_dir}/main_potts" invalid-output-root
fi
if [[ "${mode}" == "gpu" || "${mode}" == "all" || "${mode}" == "checkpoint" ]]; then
    if [[ "${mode}" == "checkpoint" ]]; then
        run_suite "${build_dir}/mcpa_potts_cuda_tests" "checkpoint"
    else
        run_suite "${build_dir}/mcpa_potts_cuda_tests"
    fi
    for heat in 0 1; do
        if [[ "${heat}" == 0 ]]; then direction="cooling"; else direction="heating"; fi
        if [[ "${mode}" != "checkpoint" ]]; then
            run_case "Potts ${direction} three-file output contract" \
                bash "${repo_dir}/tests/test_output_process.sh" \
                "${build_dir}/main_potts" \
                "${build_dir}/mcpa_potts_cuda_tests" "${heat}"
        fi
        run_case "Killed Potts ${direction} checkpoint resumes byte-identical outputs" \
            bash "${repo_dir}/tests/test_checkpoint_process.sh" \
            "${build_dir}/main_potts" "${build_dir}/mcpa_potts_cuda_tests" "${heat}"
    done
fi
