#!/usr/bin/env bash
set -euo pipefail

main_exe="${1:?main_potts executable is required}"
gpu_probe="${2:?Potts CUDA test executable is required}"
heat="${3:?heat direction is required}"

set +e
"${gpu_probe}" \
    '--test=CUDA initialization and tracked energies match independent oracles' \
    >/dev/null 2>&1
probe_status=$?
set -e
if (( probe_status == 77 )); then
    echo "SKIP: Potts output process test needs an accessible NVIDIA GPU"
    exit 77
elif (( probe_status != 0 )); then
    echo "FAIL: Potts CUDA probe returned ${probe_status}" >&2
    exit 1
fi
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/mcpa-potts-output.XXXXXX")"
trap 'rm -rf -- "${work_dir}"' EXIT

case "${heat}" in
    0) heating=""; direction="cooling" ;;
    1) heating="Heating"; direction="heating" ;;
    *) echo "FAIL: heat must be 0 or 1" >&2; exit 2 ;;
esac

run_cap() {
    local cap="$1"
    local seed="$2"
    local root="${work_dir}/cap_${cap}"
    local name="2DPotts_v3${heating}_q2_N4_R4_nSteps1_run${seed}"
    MCPA_OUTPUT_ROOT="${root}" MCPA_DETAILED_CAP="${cap}" \
        MCPA_DETERMINISTIC_TIMINGS=1 timeout 20s "${main_exe}" \
        "${seed}" 2 1 4 1 2 "${heat}" none >"${work_dir}/cap_${cap}.log" 2>&1

    mapfile -t files < <(find "${root}" -type f -printf '%f\n' | sort)
    [[ "${#files[@]}" == 3 ]]
    [[ "${files[0]}" == "${name}_agg_stats.txt" ]]
    [[ "${files[1]}" == "${name}_detailed_stats.txt" ]]
    [[ "${files[2]}" == "${name}_main.txt" ]]
    local main="${root}/2DPotts/${name}_main.txt"
    local agg="${root}/2DPotts/${name}_agg_stats.txt"
    local detailed="${root}/2DPotts/${name}_detailed_stats.txt"
    awk 'NR==1 {n=NF; next} NF!=n {exit 1} END {exit NR<2}' "${main}"
    awk 'NR==1 {n=NF; next} NF!=n {exit 1} END {exit NR<2}' "${agg}"
    awk 'NR==1 {n=NF; next} NF!=n {exit 1}' "${detailed}"
    awk 'NR==1 {next} $13!="NA" {seen++} END {exit seen!=1}' "${main}"
    [[ "$(grep -c 'terminal_full_cull' "${main}")" == 1 ]]
    [[ "$(grep -c 'no_next_shell' "${main}" || true)" == 0 ]]
    paste <(tail -n +2 "${main}") <(tail -n +2 "${agg}") \
        | awk '$1!=$21 || $4!=$22 {exit 1} END {exit NR<1}'
    [[ "$(find "${root}" -type f \( -name '*time.txt' -o -name '*.bin' \) | wc -l)" == 0 ]]
    awk -v cap="${cap}" '
        FNR==NR {if (FNR>1) population[$1]=$2; next}
        FNR>1 {if (!($1 in population)) exit 1; count[$1]++}
        END {
            for (energy in population) {
                if (cap < 0) expected = population[energy];
                else if (cap < population[energy]) expected = cap;
                else expected = population[energy];
                if (count[energy]+0 != expected) exit 1;
            }
        }' "${agg}" "${detailed}"
}

run_cap 0 $((700 + heat))
run_cap 2 $((710 + heat))
run_cap -1 $((720 + heat))
echo "PASS: Potts ${direction} v3 files join on exact shell subsets and cap semantics"
