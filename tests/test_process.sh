#!/usr/bin/env bash
set -euo pipefail

binary="$1"
contract="$2"

require_rejected() {
    local output
    if output="$(${binary} "$@" 2>&1)"; then
        echo "unexpected success: ${output}" >&2
        return 1
    fi
    if [[ "${output}" == *GPUassert* || "${output}" == *"memory estimate"* ]]; then
        echo "invalid input reached CUDA setup: ${output}" >&2
        return 1
    fi
}

case "${contract}" in
    invalid-parameters)
        require_rejected 1 1 1 1 1 2 0
        require_rejected 1 3 1 1 1 5 0
        require_rejected 1 50000 1 1 1 2 0
        require_rejected 1 3 2147483647 2 1 2 0
        require_rejected nope 3 1 1 1 2 0
        MCPA_DETAILED_CAP=-2 require_rejected 1 3 1 1 1 2 0
        MCPA_DETAILED_CAP=bad require_rejected 1 3 1 1 1 2 0
        echo "PASS: Invalid Potts parameters fail before CUDA setup"
        ;;
    output-directories)
        temp_dir="$(mktemp -d)"
        trap 'rm -rf "${temp_dir}"' EXIT
        output_root="${temp_dir}/nested/output"
        MCPA_OUTPUT_ROOT="${output_root}" "${binary}" 1 2 1 1 1 2 0 \
            >"${temp_dir}/run.log" 2>&1 || true
        [[ -d "${output_root}/2DPotts" ]]
        [[ ! -e "${output_root}/spin_samples" ]]
        echo "PASS: Potts output directories are created before CUDA setup"
        ;;
    invalid-output-root)
        temp_dir="$(mktemp -d)"
        trap 'rm -rf "${temp_dir}"' EXIT
        output_root="${temp_dir}/not-a-directory"
        : >"${output_root}"
        if MCPA_OUTPUT_ROOT="${output_root}" "${binary}" 1 2 1 1 1 2 0 \
                >"${temp_dir}/run.log" 2>&1; then
            echo "unexpected success with a file as output root" >&2
            exit 1
        fi
        grep -q "Could not create output directory" "${temp_dir}/run.log"
        if grep -qE "GPUassert|memory estimate" "${temp_dir}/run.log"; then
            echo "invalid output root reached CUDA setup" >&2
            exit 1
        fi
        echo "PASS: Invalid Potts output root fails before CUDA setup"
        ;;
    *)
        echo "Unknown process contract: ${contract}" >&2
        exit 2
        ;;
esac
