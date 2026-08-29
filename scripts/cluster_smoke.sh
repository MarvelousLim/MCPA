#!/usr/bin/env bash
# Explicit short Slurm smoke test. Running without --submit is a dry run.

#SBATCH --job-name=bw_smoke
#SBATCH --account=proj_1793
#SBATCH --gpus=1
#SBATCH --partition=rocky
#SBATCH --time=00:10:00
#SBATCH --output=bw_smoke_%j.log

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
        echo "Run from the baxter_wu repository root, or set MCPA_REPO_DIR." >&2
        return 1
    fi
    printf '%s\n' "${candidate}"
}

REPO_DIR="$(resolve_repo_dir)"
SCRIPT_PATH="${REPO_DIR}/scripts/cluster_smoke.sh"
LOG_PATTERN="${REPO_DIR}/build/bw_smoke_%j.log"
SLURM_ACCOUNT="${MCPA_SLURM_ACCOUNT:-proj_1793}"
SMOKE_WORK_DIR=""

cleanup_smoke_work_dir() {
    [[ -z "${SMOKE_WORK_DIR}" ]] || rm -rf -- "${SMOKE_WORK_DIR}"
}

submit_smoke() {
    command -v sbatch >/dev/null 2>&1 || {
        echo "ERROR: sbatch is unavailable; run this on the cluster login node." >&2
        exit 1
    }
    mkdir -p "${REPO_DIR}/build"
    echo "Submitting one short GPU smoke job (L=6, R=512, nSteps=1)."
    local submit_rc=0 job_ref job_id log_file
    job_ref="$(sbatch --parsable --wait --account="${SLURM_ACCOUNT}" \
        --export="ALL,MCPA_REPO_DIR=${REPO_DIR}" \
        --output="${LOG_PATTERN}" "${SCRIPT_PATH}" --worker)" || submit_rc=$?
    job_id="${job_ref%%;*}"
    log_file="${REPO_DIR}/build/bw_smoke_${job_id}.log"
    [[ ! -f "${log_file}" ]] || cat "${log_file}"
    if (( submit_rc == 0 )) && [[ -f "${log_file}" ]] &&
       grep -q '^SMOKE PASS:' "${log_file}"; then
        echo "PASS: cluster smoke job ${job_id}; log: ${log_file}"
        return 0
    fi
    echo "FAIL: cluster smoke job ${job_id:-unknown} (sbatch status ${submit_rc}); inspect: ${log_file}" >&2
    return 1
}

run_worker() {
    if command -v module >/dev/null 2>&1; then module load CUDA/12.4; fi
    bash "${REPO_DIR}/scripts/compile.sh"

    local exe main_file agg_file detail_file
    exe="${REPO_DIR}/build/2DBaxterWu.exe"
    SMOKE_WORK_DIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/bw-smoke.XXXXXX")"
    trap cleanup_smoke_work_dir EXIT
    mkdir -p "${SMOKE_WORK_DIR}/datasets/2DBaxterWu"
    cd "${SMOKE_WORK_DIR}"

    echo "Smoke command: ${exe} 314159 6 1 512 1 0 none"
    srun "${exe}" 314159 6 1 512 1 0 none

    main_file="${SMOKE_WORK_DIR}/datasets/2DBaxterWu/2DBaxterWu_N36_R512_nSteps1_run314159_main.txt"
    agg_file="${SMOKE_WORK_DIR}/datasets/2DBaxterWu/2DBaxterWu_N36_R512_nSteps1_run314159_agg_stats.txt"
    detail_file="${SMOKE_WORK_DIR}/datasets/2DBaxterWu/2DBaxterWu_N36_R512_nSteps1_run314159_detailed_stats.txt"
    [[ -s "${main_file}" && -s "${agg_file}" && -s "${detail_file}" ]] || {
        echo "SMOKE FAIL: expected output files were not produced" >&2
        exit 1
    }
    grep -q 'culling_factor_full_precision' "${main_file}" || {
        echo "SMOKE FAIL: main output lacks exact culling" >&2; exit 1;
    }
    grep -q 'family_effective_shannon' "${main_file}" || {
        echo "SMOKE FAIL: main output lacks genealogy" >&2; exit 1;
    }
    awk -F '\t' '
        NR == 1 {
            for (i = 1; i <= NF; ++i) if ($i == "gpu_name") gpu_column = i
            next
        }
        NR == 2 { exit !(gpu_column && $gpu_column != "" && $gpu_column != "NA") }
    ' "${main_file}" || {
        echo "SMOKE FAIL: main output lacks its first-row GPU record" >&2; exit 1;
    }
    grep -q 'order_sf_k3' "${agg_file}" || {
        echo "SMOKE FAIL: aggregate output lacks correlation-length fields" >&2; exit 1;
    }
    grep -q 'family_cov_mean_order_sf0_kmin' "${agg_file}" || {
        echo "SMOKE FAIL: aggregate output lacks family-cluster fields" >&2; exit 1;
    }
    echo "SMOKE PASS: executable completed; three files, GPU record, and diagnostic schema verified"
}

case "${1:-}" in
    --submit) submit_smoke ;;
    --worker) run_worker ;;
    --paths)
        echo "Repository: ${REPO_DIR}"
        echo "Executable: ${REPO_DIR}/build/2DBaxterWu.exe"
        echo "Smoke log:  ${LOG_PATTERN}"
        ;;
    ""|--dry-run)
        echo "No job submitted. To run one short smoke job explicitly:"
        echo "  ${SCRIPT_PATH} --submit"
        ;;
    *)
        echo "Usage: $0 [--dry-run|--submit|--paths]" >&2
        exit 2
        ;;
esac
