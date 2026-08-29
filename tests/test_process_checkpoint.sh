#!/usr/bin/env bash
set -euo pipefail

main_binary="$1"
gpu_probe="$2"

set +e
"${gpu_probe}" --test="Initialized GPU population has valid spins and exact energy parts" \
    >/dev/null 2>&1
probe_status=$?
set -e
if [[ ${probe_status} -eq 77 ]]; then
    echo "SKIP: process checkpoint test needs an accessible NVIDIA GPU"
    exit 77
fi
if [[ ${probe_status} -ne 0 ]]; then
    echo "FAIL: CUDA probe failed" >&2
    exit "${probe_status}"
fi

test_root="$(mktemp -d /tmp/mcpa-bc-process-checkpoint-XXXXXX)"
child_pid=""
cleanup() {
    status=$?
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
        kill -9 "${child_pid}" 2>/dev/null || true
        wait "${child_pid}" 2>/dev/null || true
    fi
    if [[ ${status} -ne 0 ]]; then
        for diagnostic_log in "${test_root}/baseline/run.log" \
                              "${test_root}/interrupted/run.log"; do
            if [[ -f "${diagnostic_log}" ]]; then
                echo "--- ${diagnostic_log} ---" >&2
                cat "${diagnostic_log}" >&2
            fi
        done
    fi
    rm -rf -- "${test_root}"
    return "${status}"
}
trap cleanup EXIT

mkdir -p "${test_root}/baseline" "${test_root}/interrupted"
run_args=(941 8 2 32 1 0 1.96 checkpoints 0 -1)

(
    cd "${test_root}/baseline"
    "${main_binary}" "${run_args[@]}" >run.log 2>&1
)

(
    cd "${test_root}/interrupted"
    exec "${main_binary}" "${run_args[@]}" >run.log 2>&1
) &
child_pid=$!

killed=0
for ((attempt = 0; attempt < 2000; ++attempt)); do
    checkpoint_files=("${test_root}/interrupted"/checkpoints/*.bin)
    if [[ -e "${checkpoint_files[0]}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
        kill -9 "${child_pid}"
        wait "${child_pid}" 2>/dev/null || true
        child_pid=""
        killed=1
        break
    fi
    if ! kill -0 "${child_pid}" 2>/dev/null; then
        wait "${child_pid}" || true
        child_pid=""
        break
    fi
    sleep 0.005
done
if [[ ${killed} -ne 1 ]]; then
    echo "FAIL: run completed before a checkpoint could be killed" >&2
    exit 1
fi

(
    cd "${test_root}/interrupted"
    "${main_binary}" "${run_args[@]}" >>run.log 2>&1
)

baseline_outputs=("${test_root}/baseline"/datasets/2DBlumeEnergyParts/*.txt)
interrupted_outputs=("${test_root}/interrupted"/datasets/2DBlumeEnergyParts/*.txt)
if [[ ${#baseline_outputs[@]} -ne 3 || ${#interrupted_outputs[@]} -ne 3 ]]; then
    echo "FAIL: expected three output files from each run" >&2
    exit 1
fi
for baseline_file in "${baseline_outputs[@]}"; do
    file_name="$(basename "${baseline_file}")"
    interrupted_file="${test_root}/interrupted/datasets/2DBlumeEnergyParts/${file_name}"
    if [[ "${file_name}" == *_main.txt ]]; then
        awk -F '\t' '
            NF != 18 {exit 1}
            NR > 1 && $9 < 0 {exit 1}
            NR > 1 && $12 != "NA" {metadata_rows++}
            NR == 2 && ($15 <= 0 || $16 <= 0 || $17 <= 0 || $18 <= 0) {exit 1}
            NR == 2 && ($15 < $16 || $15 < $17) {exit 1}
            NR > 2 && ($12 != "NA" || $13 != "NA" || $14 != "NA" || $15 != "NA" || $16 != "NA" || $17 != "NA" || $18 != "NA") {exit 1}
            END {exit metadata_rows == 1 ? 0 : 1}
        ' "${baseline_file}"
        awk -F '\t' 'BEGIN {OFS="\t"} NR > 1 {$9="TIMING"} {print}' \
            "${baseline_file}" >"${test_root}/baseline-main-normalized.txt"
        awk -F '\t' 'BEGIN {OFS="\t"} NR > 1 {$9="TIMING"} {print}' \
            "${interrupted_file}" >"${test_root}/interrupted-main-normalized.txt"
        cmp --silent "${test_root}/baseline-main-normalized.txt" \
            "${test_root}/interrupted-main-normalized.txt"
    else
        awk -F '\t' 'NF != 11 {exit 1}' "${baseline_file}"
        cmp --silent "${baseline_file}" "${interrupted_file}"
    fi
done

snapshot_dir="${test_root}/completed-snapshot"
mkdir -p "${snapshot_dir}"
for interrupted_file in "${interrupted_outputs[@]}"; do
    cp "${interrupted_file}" "${snapshot_dir}/$(basename "${interrupted_file}")"
done

(
    cd "${test_root}/interrupted"
    "${main_binary}" "${run_args[@]}" >>run.log 2>&1
)
for interrupted_file in "${interrupted_outputs[@]}"; do
    cmp --silent "${interrupted_file}" \
        "${snapshot_dir}/$(basename "${interrupted_file}")"
done

done_files=("${test_root}/interrupted"/checkpoints/*.done)
current_files=("${test_root}/interrupted"/checkpoints/*.bin)
if [[ ${#done_files[@]} -ne 1 || ! -e "${done_files[0]}"
      || ${#current_files[@]} -lt 1 || ! -e "${current_files[0]}" ]]; then
    echo "FAIL: expected completed marker and checkpoint candidates" >&2
    exit 1
fi
rm -f -- "${done_files[0]}"
for checkpoint_file in "${current_files[@]}"; do
    truncate -s 1 "${checkpoint_file}"
done
if (
    cd "${test_root}/interrupted"
    "${main_binary}" "${run_args[@]}" >>run.log 2>&1
); then
    echo "FAIL: invalid checkpoint set started a fresh run" >&2
    exit 1
fi
for interrupted_file in "${interrupted_outputs[@]}"; do
    cmp --silent "${interrupted_file}" \
        "${snapshot_dir}/$(basename "${interrupted_file}")"
done

echo "PASS: restart preserves physics columns, done is a no-op, invalid state preserves outputs"
