#!/usr/bin/env bash
# Checkpointed HSE size campaign: one human-selected size and population, two
# independent seeds, and both directions (4 tasks). The current default is the
# first production size, L=162:
#   sbatch --export=ALL,MCPA_R=32768 scripts/hse_meta_2DBaxterWu.sh
# Later stages reuse the same checkpoint contract, for example:
#   sbatch --time=7-00:00:00 --export=ALL,MCPA_L=243,MCPA_R=32768 scripts/hse_meta_2DBaxterWu.sh
# For L=363 use the documented high-memory-node safety constraint:
#   sbatch --constraint=type_e --time=7-00:00:00 --export=ALL,MCPA_L=363,MCPA_R=32768 scripts/hse_meta_2DBaxterWu.sh

#SBATCH --job-name=bw_scale
#SBATCH --account=proj_1793
#SBATCH --gpus=1
#SBATCH --mail-user=limen2402@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3
#SBATCH --partition=rocky
#SBATCH --time=2-00:00:00
#SBATCH --output=bw_scale_%A_%a.out

set -euo pipefail

SOURCE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

resolve_repo_dir() {
    local candidate
    if [[ -n "${MCPA_REPO_DIR:-}" ]]; then
        candidate="${MCPA_REPO_DIR}"
    elif [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        candidate="${SLURM_SUBMIT_DIR}"
    else
        candidate="${SOURCE_SCRIPT_DIR}/.."
    fi
    if [[ "$(basename -- "${candidate}")" == "scripts" ]]; then
        candidate="${candidate}/.."
    fi
    candidate="$(cd -- "${candidate}" 2>/dev/null && pwd)" || {
        echo "ERROR: repository anchor does not exist: ${candidate}" >&2
        return 1
    }
    if [[ ! -f "${candidate}/CMakeLists.txt" ||
          ! -f "${candidate}/main/main.cpp" ||
          ! -f "${candidate}/scripts/compile.sh" ]]; then
        echo "ERROR: expected CMakeLists.txt, main/, lib/, and scripts/ under: ${candidate}" >&2
        echo "Submit from the repository root or set MCPA_REPO_DIR." >&2
        return 1
    fi
    printf '%s\n' "${candidate}"
}

require_positive_integer() {
    local name="$1" value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${name} must be a positive integer, got '${value}'" >&2
        return 1
    fi
}

map_task() {
    local task_id="$1"
    if [[ ! "${task_id}" =~ ^[0-9]+$ ]] || (( task_id < 0 || task_id > 3 )); then
        echo "ERROR: array task id must be in 0..3, got '${task_id}'" >&2
        return 1
    fi
    SEED=$((task_id / 2 + 1))
    HEAT=$((task_id % 2))
    if (( HEAT == 1 )); then DIRECTION=heating; else DIRECTION=cooling; fi
}

REPO_DIR="$(resolve_repo_dir)"
EXE="${MCPA_EXE:-${REPO_DIR}/build/2DBaxterWu.exe}"
CHECKPOINT_HOURS="${MCPA_CHECKPOINT_HOURS:-0.25}"
DETAILED_LIMIT="${MCPA_DETAILED_LIMIT:-1000}"
THREADS=512
NSTEPS=10

if [[ -z "${MCPA_R:-}" ]]; then
    echo "ERROR: MCPA_R is required; submit with --export=ALL,MCPA_R=<population>." >&2
    exit 2
fi
R="${MCPA_R}"
require_positive_integer MCPA_R "${R}"
L="${MCPA_L:-162}"
require_positive_integer MCPA_L "${L}"
if (( L % 3 != 0 || L > 363 )); then
    echo "ERROR: MCPA_L=${L} must be divisible by 3 and no larger than the approved campaign ceiling 363" >&2
    exit 2
fi
if (( R > 2147483647 )); then
    echo "ERROR: MCPA_R must fit a signed 32-bit integer" >&2
    exit 2
fi
if (( R % THREADS != 0 )); then
    echo "ERROR: MCPA_R=${R} must be divisible by the pilot thread count ${THREADS}" >&2
    exit 2
fi
if [[ ! "${CHECKPOINT_HOURS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: MCPA_CHECKPOINT_HOURS must be a nonnegative decimal" >&2
    exit 2
fi
if [[ ! "${DETAILED_LIMIT}" =~ ^(-1|0|[1-9][0-9]*)$ ]] ||
   (( DETAILED_LIMIT > 2147483647 )); then
    echo "ERROR: MCPA_DETAILED_LIMIT must be -1, 0, or a positive signed 32-bit integer" >&2
    exit 2
fi
BLOCKS=$((R / THREADS))
CAMPAIGN_DIR="${MCPA_CAMPAIGN_DIR:-${REPO_DIR}/experiments/hse_bw_L${L}_R${R}_steps${NSTEPS}_v1}"

if [[ "${1:-}" == "--paths" ]]; then
    echo "Repository: ${REPO_DIR}"
    echo "Executable: ${EXE}"
    echo "Campaign:   ${CAMPAIGN_DIR}"
    exit 0
fi

if [[ "${1:-}" == "--map" ]]; then
    printf 'task\tL\tR\tnSteps\tseed\theat\tdirection\tdetailed_limit\n'
    for task_id in {0..3}; do
        map_task "${task_id}"
        printf '%d\t%d\t%d\t%d\t%d\t%d\t%s\t%d\n' \
            "${task_id}" "${L}" "${R}" "${NSTEPS}" "${SEED}" "${HEAT}" \
            "${DIRECTION}" "${DETAILED_LIMIT}"
    done
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--paths|--map]" >&2
    exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?Run through sbatch, or use --map for local validation}"
map_task "${TASK_ID}"

N=$((L * L))
RUN_DIR="${CAMPAIGN_DIR}/runs/L${L}_R${R}_steps${NSTEPS}_seed${SEED}_${DIRECTION}_detail${DETAILED_LIMIT}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
if (( HEAT == 1 )); then HEATING_NAME=Heating; else HEATING_NAME=; fi
BASE="2DBaxterWu${HEATING_NAME}_N${N}_R${R}_nSteps${NSTEPS}_run${SEED}"
CHECKPOINT_BASE="${CHECKPOINT_DIR}/${BASE}_chk"
OUTPUT_BASE="${RUN_DIR}/datasets/2DBaxterWu/${BASE}"

if [[ -e "${CHECKPOINT_BASE}.done" ]]; then
    for suffix in _main.txt _agg_stats.txt _detailed_stats.txt; do
        if [[ ! -s "${OUTPUT_BASE}${suffix}" ]]; then
            echo "ERROR: .done exists but required output is missing or empty: ${OUTPUT_BASE}${suffix}" >&2
            exit 4
        fi
    done
    echo "Baxter-Wu task ${TASK_ID} is already complete; .done exists. No srun and no output changes."
    exit 0
fi

has_checkpoint=0
for suffix in .bin .prev.bin .tmp; do
    if [[ -e "${CHECKPOINT_BASE}${suffix}" ]]; then has_checkpoint=1; fi
done
has_output=0
for suffix in _main.txt _agg_stats.txt _detailed_stats.txt; do
    if [[ -e "${OUTPUT_BASE}${suffix}" ]]; then has_output=1; fi
done
if (( has_checkpoint == 0 && has_output == 1 )) &&
   [[ "${MCPA_ALLOW_FRESH_RESTART:-0}" != "1" ]]; then
    echo "ERROR: outputs exist but no checkpoint generation is available." >&2
    echo "Refusing to truncate them. Inspect/remove the run directory, or explicitly set MCPA_ALLOW_FRESH_RESTART=1." >&2
    exit 3
fi

if command -v module >/dev/null 2>&1; then
    # compile.sh links with the GNU 12 runtime. Slurm starts each array task
    # from the site-default GNU 8 environment, so repeat the toolchain switch
    # inside the batch job instead of relying on the submit shell's modules.
    if module is-loaded gnu8 >/dev/null 2>&1 &&
       module is-avail gnu12/12.1 >/dev/null 2>&1; then
        echo "Switching task compiler runtime: gnu8 -> gnu12/12.1"
        module swap gnu8 gnu12/12.1
    fi
    echo "Loading task CUDA runtime: CUDA/12.4"
    module load CUDA/12.4
fi
[[ -x "${EXE}" ]] || { echo "ERROR: executable not found: ${EXE}" >&2; exit 1; }

RUNTIME_DEPENDENCIES="$(ldd "${EXE}" 2>&1 || true)"
if grep -Eq 'not found|required by' <<<"${RUNTIME_DEPENDENCIES}"; then
    echo "ERROR: executable runtime dependencies are not satisfied after loading the task modules:" >&2
    printf '%s\n' "${RUNTIME_DEPENDENCIES}" >&2
    exit 1
fi

echo "Baxter-Wu size-campaign task ${TASK_ID}: L=${L} R=${R} nSteps=${NSTEPS} seed=${SEED} heat=${HEAT}"
echo "Direction: ${DIRECTION}; isolated run directory: ${RUN_DIR}"
echo "Checkpoint interval: ${CHECKPOINT_HOURS} h; directory: ${CHECKPOINT_DIR}"
echo "Detailed output: deterministic hash sample capped at ${DETAILED_LIMIT} replicas per shell"
if (( has_checkpoint == 1 )); then
    echo "Rerun mode: checkpoint candidate found; main_bw will validate it, resume and truncate to saved offsets, or fail without changing outputs."
elif (( has_output == 1 )); then
    echo "Rerun mode: explicit MCPA_ALLOW_FRESH_RESTART=1; existing outputs will be replaced by a fresh run."
else
    echo "Run mode: fresh; no checkpoint or outputs exist."
fi
printf 'Command: srun %q %q %q %q %q %q %q %q %q %q\n' \
    "${EXE}" "${SEED}" "${L}" "${BLOCKS}" "${THREADS}" "${NSTEPS}" "${HEAT}" \
    "${CHECKPOINT_DIR}" "${CHECKPOINT_HOURS}" "${DETAILED_LIMIT}"

mkdir -p "${RUN_DIR}/datasets/2DBaxterWu"
cd "${RUN_DIR}"
srun "${EXE}" "${SEED}" "${L}" "${BLOCKS}" "${THREADS}" "${NSTEPS}" "${HEAT}" \
    "${CHECKPOINT_DIR}" "${CHECKPOINT_HOURS}" "${DETAILED_LIMIT}"
