#!/usr/bin/env bash
set -euo pipefail

main_exe="${1:?main_potts executable is required}"
gpu_probe="${2:?Potts CUDA test executable is required}"
heat="${3:-0}"
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
"${gpu_probe}" \
    '--test=CUDA initialization and tracked energies match independent oracles' \
    >/dev/null 2>&1
probe_status=$?
set -e
if (( probe_status == 77 )); then
    echo "SKIP: Potts checkpoint process test needs an accessible NVIDIA GPU"
    exit 77
elif (( probe_status != 0 )); then
    echo "FAIL: Potts CUDA probe returned ${probe_status}" >&2
    exit 1
fi

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/mcpa-potts-process-checkpoint.XXXXXX")"
mkdir -p "${work_dir}/baseline" "${work_dir}/resumed"

seed=821
L=10
blocks=2
threads=64
nsteps=5
q=3
N=$((L * L))
R=$((blocks * threads))
if [[ "${heat}" == 1 ]]; then heating="Heating"; else heating=""; fi
name="2DPotts_v3${heating}_q${q}_N${N}_R${R}_nSteps${nsteps}_run${seed}"

baseline_root="${work_dir}/baseline/datasets"
resumed_root="${work_dir}/resumed/datasets"
cd "${work_dir}/baseline"
MCPA_OUTPUT_ROOT="${baseline_root}" MCPA_DETAILED_CAP=5 \
    MCPA_DETERMINISTIC_TIMINGS=1 timeout 30s "${main_exe}" \
    "${seed}" "${L}" "${blocks}" "${threads}" "${nsteps}" "${q}" "${heat}" \
    none >baseline.log 2>&1

cd "${work_dir}/resumed"
checkpoint_dir="${work_dir}/checkpoints"
MCPA_OUTPUT_ROOT="${resumed_root}" MCPA_DETAILED_CAP=5 \
    MCPA_DETERMINISTIC_TIMINGS=1 MCPA_TEST_PAUSE_AFTER_CHECKPOINT_MS=100 \
    "${main_exe}" \
    "${seed}" "${L}" "${blocks}" "${threads}" "${nsteps}" "${q}" "${heat}" \
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
    echo "FAIL: no committed Potts checkpoint observed before exit" >&2
    tail -n 30 interrupted.log >&2 || true
    exit 1
fi

kill -KILL "${run_pid}" 2>/dev/null || true
wait "${run_pid}" 2>/dev/null || true
run_pid=""
if [[ -e "${checkpoint_dir}/${name}_chk.done" ]]; then
    echo "FAIL: interrupted Potts process wrote a done marker" >&2
    exit 1
fi

MCPA_OUTPUT_ROOT="${resumed_root}" MCPA_DETAILED_CAP=5 \
    MCPA_DETERMINISTIC_TIMINGS=1 timeout 30s "${main_exe}" \
    "${seed}" "${L}" "${blocks}" "${threads}" "${nsteps}" "${q}" "${heat}" \
    "${checkpoint_dir}" 1 >continued.log 2>&1
test -s "${checkpoint_dir}/${name}_chk.done"

baseline_prefix="${baseline_root}/2DPotts/${name}"
resumed_prefix="${resumed_root}/2DPotts/${name}"
for suffix in _main.txt _agg_stats.txt _detailed_stats.txt; do
    cmp "${baseline_prefix}${suffix}" "${resumed_prefix}${suffix}"
done

snapshot_root="${work_dir}/completed-snapshot"
mkdir -p "${snapshot_root}"
cp -a "${resumed_root}/2DPotts" "${snapshot_root}/"

MCPA_OUTPUT_ROOT="${resumed_root}" MCPA_DETAILED_CAP=5 \
    MCPA_DETERMINISTIC_TIMINGS=1 timeout 10s "${main_exe}" \
    "${seed}" "${L}" "${blocks}" "${threads}" "${nsteps}" "${q}" "${heat}" \
    "${checkpoint_dir}" 1 >done-rerun.log 2>&1
diff -r "${snapshot_root}/2DPotts" "${resumed_root}/2DPotts"

done_file="${checkpoint_dir}/${name}_chk.done"
rm -f -- "${done_file}"
for checkpoint_candidate in "${checkpoint_dir}/${name}_chk.bin" \
                            "${checkpoint_dir}/${name}_chk.prev.bin" \
                            "${checkpoint_dir}/${name}_chk.tmp"; do
    if [[ -e "${checkpoint_candidate}" ]]; then
        truncate -s 1 "${checkpoint_candidate}"
    fi
done
if MCPA_OUTPUT_ROOT="${resumed_root}" MCPA_DETAILED_CAP=5 \
        MCPA_DETERMINISTIC_TIMINGS=1 timeout 10s "${main_exe}" \
        "${seed}" "${L}" "${blocks}" "${threads}" "${nsteps}" "${q}" "${heat}" \
        "${checkpoint_dir}" 1 >invalid-rerun.log 2>&1; then
    echo "FAIL: invalid Potts checkpoint set started a fresh run" >&2
    exit 1
fi
diff -r "${snapshot_root}/2DPotts" "${resumed_root}/2DPotts"

echo "PASS: Potts ${heating:-cooling} restart is identical, done/invalid preserve three outputs"
