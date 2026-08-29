#!/usr/bin/env bash
set -euo pipefail

experiment="${1:-}"
variant="${2:-candidate}"
preset="${3:-smoke}"
regime="${4:-open}"
direction="${5:-cooling}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${repo_dir}/optimization/results"

if [[ ! "${experiment}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Usage: ./optimization/run.sh EXPERIMENT [candidate|reference|both] [smoke|production|working-set] [open|frozen|mixed] [cooling|heating]" >&2
    exit 2
fi
case "${variant}" in candidate|reference|both) ;;
    *) echo "ERROR: variant must be candidate, reference, or both" >&2; exit 2 ;;
esac
case "${preset}" in smoke|production|working-set) ;;
    *) echo "ERROR: unknown preset: ${preset}" >&2; exit 2 ;;
esac
case "${regime}" in open|frozen|mixed) ;;
    *) echo "ERROR: regime must be open, frozen, or mixed" >&2; exit 2 ;;
esac
case "${direction}" in cooling|heating) ;;
    *) echo "ERROR: direction must be cooling or heating" >&2; exit 2 ;;
esac

mkdir -p "${result_dir}"
result_file="${result_dir}/${experiment}.jsonl"
timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "${repo_dir}" rev-parse --short HEAD 2>/dev/null || printf unknown)"

echo "Experiment ${experiment}: ${variant}, ${preset}, ${regime}, ${direction}, commit ${commit}"
echo "Correctness check is enabled; timing starts only if it passes."
benchmark_args=(--variant "${variant}" --regime "${regime}" --json)
if [[ "${direction}" == "heating" ]]; then benchmark_args+=(--heat); fi
output="$("${repo_dir}/benchmark.sh" "${preset}" "${benchmark_args[@]}")"
printf '%s\n' "${output}"
while IFS= read -r line; do
    [[ "${line}" == \{*\} ]] && printf '%s\n' "${line}" >> "${result_file}"
done <<< "${output}"
printf '| %s | %s | %s | %s | %s | %s | %s | `%s` |\n' \
    "${timestamp}" "${experiment}" "${variant}" "${preset}" "${regime}" "${direction}" "${commit}" \
    "results/${experiment}.jsonl" >> "${result_dir}/index.md"
echo "Recorded ${result_file}"
