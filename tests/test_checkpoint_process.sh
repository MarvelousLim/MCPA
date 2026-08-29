#!/usr/bin/env bash
# Cluster-style integration check: kill a real MCPA process after a committed
# checkpoint, resume it, and compare all deterministic outputs with a fresh run.

set -euo pipefail

main_exe="${1:?main_bw executable is required}"
cuda_test_exe="${2:?mcpa_cuda_tests executable is required}"
run_pid=""
work_dir=""

cleanup() {
    if [[ -n "${run_pid}" ]] && kill -0 "${run_pid}" 2>/dev/null; then
        kill -KILL "${run_pid}" 2>/dev/null || true
        wait "${run_pid}" 2>/dev/null || true
    fi
    if [[ -n "${work_dir}" && -d "${work_dir}" ]]; then
        rm -rf -- "${work_dir}"
    fi
}
trap cleanup EXIT INT TERM

set +e
"${cuda_test_exe}" \
    '--test-case=GPU memory allocation and round-trip transfer succeed' \
    --no-version --no-intro >/dev/null 2>&1
gpu_status=$?
set -e
if (( gpu_status == 77 )); then
    echo "SKIP: process checkpoint test needs an accessible NVIDIA GPU"
    exit 77
elif (( gpu_status != 0 )); then
    echo "FAIL: CUDA accessibility probe failed with status ${gpu_status}" >&2
    exit "${gpu_status}"
fi

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/mcpa-process-checkpoint.XXXXXX")"
mkdir -p "${work_dir}/baseline" "${work_dir}/resumed"

seed=813
L=18
blocks=8
threads=512
nsteps=5
heat=0
N=$((L * L))
R=$((blocks * threads))
name="2DBaxterWu_N${N}_R${R}_nSteps${nsteps}_run${seed}"

cd "${work_dir}/baseline"
timeout 30s "${main_exe}" \
    "${seed}" "${L}" "${blocks}" "${threads}" "${nsteps}" "${heat}" none \
    >baseline.log

cd "${work_dir}/resumed"
checkpoint_dir="${work_dir}/checkpoints"
"${main_exe}" \
    "${seed}" "${L}" "${blocks}" "${threads}" "${nsteps}" "${heat}" \
    "${checkpoint_dir}" 0 >interrupted.log 2>&1 &
run_pid=$!
checkpoint_file="${checkpoint_dir}/${name}_chk.bin"

found=0
for ((attempt = 0; attempt < 500; ++attempt)); do
    if [[ -s "${checkpoint_file}" ]]; then
        found=1
        break
    fi
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
[[ ! -e "${checkpoint_dir}/${name}_chk.done" ]] || {
    echo "FAIL: interrupted process unexpectedly marked the run complete" >&2
    exit 1
}

timeout 30s "${main_exe}" \
    "${seed}" "${L}" "${blocks}" "${threads}" "${nsteps}" "${heat}" \
    "${checkpoint_dir}" 24 >resumed.log

baseline_prefix="${work_dir}/baseline/datasets/2DBaxterWu/${name}"
resumed_prefix="${work_dir}/resumed/datasets/2DBaxterWu/${name}"

# Equilibration time and first-row GPU resource metadata are measurements, not
# trajectory state.  Every other main column and both complete statistics files
# must match byte-for-byte.
cut --complement -f6,12-18 "${baseline_prefix}_main.txt" \
    >"${work_dir}/baseline_main_physics.tsv"
cut --complement -f6,12-18 "${resumed_prefix}_main.txt" \
    >"${work_dir}/resumed_main_physics.tsv"
cmp "${work_dir}/baseline_main_physics.tsv" "${work_dir}/resumed_main_physics.tsv"
cmp "${baseline_prefix}_agg_stats.txt" "${resumed_prefix}_agg_stats.txt"
cmp "${baseline_prefix}_detailed_stats.txt" "${resumed_prefix}_detailed_stats.txt"

snapshot_dir="${work_dir}/completed-snapshot"
mkdir -p "${snapshot_dir}"
cp "${resumed_prefix}_main.txt" "${snapshot_dir}/main.txt"
cp "${resumed_prefix}_agg_stats.txt" "${snapshot_dir}/agg.txt"
cp "${resumed_prefix}_detailed_stats.txt" "${snapshot_dir}/detail.txt"

timeout 10s "${main_exe}" \
    "${seed}" "${L}" "${blocks}" "${threads}" "${nsteps}" "${heat}" \
    "${checkpoint_dir}" 24 >done-rerun.log
cmp "${snapshot_dir}/main.txt" "${resumed_prefix}_main.txt"
cmp "${snapshot_dir}/agg.txt" "${resumed_prefix}_agg_stats.txt"
cmp "${snapshot_dir}/detail.txt" "${resumed_prefix}_detailed_stats.txt"

rm -f -- "${checkpoint_dir}/${name}_chk.done"
for checkpoint_candidate in "${checkpoint_dir}/${name}_chk.bin" \
                            "${checkpoint_dir}/${name}_chk.prev.bin" \
                            "${checkpoint_dir}/${name}_chk.tmp"; do
    if [[ -e "${checkpoint_candidate}" ]]; then
        truncate -s 1 "${checkpoint_candidate}"
    fi
done
if timeout 10s "${main_exe}" \
        "${seed}" "${L}" "${blocks}" "${threads}" "${nsteps}" "${heat}" \
        "${checkpoint_dir}" 24 >invalid-rerun.log 2>&1; then
    echo "FAIL: invalid Baxter-Wu checkpoint set started a fresh run" >&2
    exit 1
fi
cmp "${snapshot_dir}/main.txt" "${resumed_prefix}_main.txt"
cmp "${snapshot_dir}/agg.txt" "${resumed_prefix}_agg_stats.txt"
cmp "${snapshot_dir}/detail.txt" "${resumed_prefix}_detailed_stats.txt"

echo "PASS: crash resume matches, done is a no-op, invalid state preserves outputs"
