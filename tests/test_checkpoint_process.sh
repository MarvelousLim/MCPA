#!/usr/bin/env bash
set -euo pipefail

main_exe="${1:?main_1d_ising executable is required}"
gpu_probe="${2:?CUDA test executable is required}"
run_pid=""
work_dir=""

cleanup() {
    if [[ -n "${run_pid}" ]] && kill -0 "${run_pid}" 2>/dev/null; then
        kill -KILL "${run_pid}" 2>/dev/null || true
        wait "${run_pid}" 2>/dev/null || true
    fi
    if [[ -n "${work_dir}" && -d "${work_dir}" ]]; then rm -rf -- "${work_dir}"; fi
}
trap cleanup EXIT INT TERM

set +e
"${gpu_probe}" --list >/dev/null 2>&1
probe_status=$?
set -e
if (( probe_status == 77 )); then
    echo "SKIP: process checkpoint test needs an accessible NVIDIA GPU"
    exit 77
elif (( probe_status != 0 )); then
    echo "FAIL: CUDA accessibility probe returned ${probe_status}" >&2
    exit 1
fi

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/mcpa-1d-process-checkpoint.XXXXXX")"
mkdir -p "${work_dir}/baseline" "${work_dir}/resumed"
seed=813 N=65 blocks=2 threads=64 nsteps=10 cap=7
R=$((blocks * threads))
name="1DIsing_N${N}_R${R}_nSteps${nsteps}_run${seed}"
suffixes=(_main.txt _agg_stats.txt _detailed_stats.txt)

cd "${work_dir}/baseline"
MCPA_DETAILED_CAP="${cap}" timeout 30s "${main_exe}" \
    "${seed}" "${N}" "${blocks}" "${threads}" "${nsteps}" none \
    >baseline.log 2>&1

cd "${work_dir}/resumed"
checkpoint_dir="${work_dir}/checkpoints"
MCPA_DETAILED_CAP="${cap}" "${main_exe}" \
    "${seed}" "${N}" "${blocks}" "${threads}" "${nsteps}" \
    "${checkpoint_dir}" 1 >interrupted.log 2>&1 &
run_pid=$!
checkpoint_file="${checkpoint_dir}/${name}_chk.bin"

found=0
for ((attempt = 0; attempt < 400; ++attempt)); do
    if [[ -s "${checkpoint_file}" ]]; then found=1; break; fi
    if ! kill -0 "${run_pid}" 2>/dev/null; then break; fi
    sleep 0.05
done
if (( found == 0 )); then
    echo "FAIL: no committed checkpoint was observed before process exit" >&2
    tail -n 20 interrupted.log >&2 || true
    exit 1
fi
kill -KILL "${run_pid}" 2>/dev/null || true
wait "${run_pid}" 2>/dev/null || true
run_pid=""
test ! -e "${checkpoint_dir}/${name}_chk.done"

resumed_prefix="${work_dir}/resumed/datasets/test/${name}"
identity_snapshot="${work_dir}/identity-snapshot"
mkdir -p "${identity_snapshot}"
for suffix in "${suffixes[@]}"; do cp "${resumed_prefix}${suffix}" "${identity_snapshot}/${suffix}"; done
if MCPA_DETAILED_CAP=8 timeout 10s "${main_exe}" \
    "${seed}" "${N}" "${blocks}" "${threads}" "${nsteps}" \
    "${checkpoint_dir}" 1 >wrong-cap.log 2>&1; then
    echo "FAIL: checkpoint accepted a different detailed cap" >&2
    exit 1
fi
for suffix in "${suffixes[@]}"; do cmp "${identity_snapshot}/${suffix}" "${resumed_prefix}${suffix}"; done

MCPA_DETAILED_CAP="${cap}" timeout 30s "${main_exe}" \
    "${seed}" "${N}" "${blocks}" "${threads}" "${nsteps}" \
    "${checkpoint_dir}" 1 >continued.log 2>&1
test -s "${checkpoint_dir}/${name}_chk.done"

baseline_prefix="${work_dir}/baseline/datasets/test/${name}"
cmp "${baseline_prefix}_agg_stats.txt" "${resumed_prefix}_agg_stats.txt"
cmp "${baseline_prefix}_detailed_stats.txt" "${resumed_prefix}_detailed_stats.txt"
# Wall time and one-time free-memory metadata are intentionally observational.
# Compare every deterministic main-table field after masking those columns.
cut -f1-7,9-13 "${baseline_prefix}_main.txt" >"${work_dir}/baseline-main-deterministic"
cut -f1-7,9-13 "${resumed_prefix}_main.txt" >"${work_dir}/resumed-main-deterministic"
cmp "${work_dir}/baseline-main-deterministic" "${work_dir}/resumed-main-deterministic"

snapshot_dir="${work_dir}/completed-snapshot"
mkdir -p "${snapshot_dir}"
for suffix in "${suffixes[@]}"; do cp "${resumed_prefix}${suffix}" "${snapshot_dir}/${suffix}"; done
MCPA_DETAILED_CAP="${cap}" timeout 10s "${main_exe}" \
    "${seed}" "${N}" "${blocks}" "${threads}" "${nsteps}" \
    "${checkpoint_dir}" 1 >done-rerun.log 2>&1
for suffix in "${suffixes[@]}"; do cmp "${snapshot_dir}/${suffix}" "${resumed_prefix}${suffix}"; done

rm -f -- "${checkpoint_dir}/${name}_chk.done"
printf 'bad-current' >"${checkpoint_dir}/${name}_chk.bin"
printf 'bad-previous' >"${checkpoint_dir}/${name}_chk.prev.bin"
printf 'bad-temporary' >"${checkpoint_dir}/${name}_chk.tmp"
if MCPA_DETAILED_CAP="${cap}" timeout 10s "${main_exe}" \
    "${seed}" "${N}" "${blocks}" "${threads}" "${nsteps}" \
    "${checkpoint_dir}" 1 >invalid-rerun.log 2>&1; then
    echo "FAIL: invalid checkpoint generations were accepted" >&2
    exit 1
fi
for suffix in "${suffixes[@]}"; do cmp "${snapshot_dir}/${suffix}" "${resumed_prefix}${suffix}"; done
test "$(find "$(dirname "${resumed_prefix}")" -maxdepth 1 -type f -name "${name}*" | wc -l)" -eq 3

echo "PASS: killed restart preserves deterministic three-file output and invalid/no-op safety"
