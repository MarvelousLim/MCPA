#!/usr/bin/env bash
set -euo pipefail

canonical="$1"
legacy="$2"
gpu_probe="$3"

set +e
"${gpu_probe}" --list >/dev/null 2>&1
probe_status=$?
set -e
if (( probe_status == 77 )); then
    echo "SKIP: CLI/output process test needs an accessible NVIDIA GPU"
    exit 77
elif (( probe_status != 0 )); then
    echo "FAIL: GPU preflight returned ${probe_status}" >&2
    exit 1
fi

process_dir="$(mktemp -d "${TMPDIR:-/tmp}/mcpa-1d-cli.XXXXXX")"
trap 'rm -rf -- "${process_dir}"' EXIT
cd "${process_dir}"

if "${canonical}" 1 2 1 1 1 >invalid.log 2>&1; then
    echo "FAIL: canonical CLI accepted N=2" >&2
    exit 1
fi
if "${canonical}" 1 5 2147483647 2 1 >overflow.log 2>&1; then
    echo "FAIL: canonical CLI accepted overflowing replica product" >&2
    exit 1
fi
if MCPA_DETAILED_CAP=-2 "${canonical}" 1 5 1 8 1 >cap-invalid.log 2>&1; then
    echo "FAIL: detailed cap accepted -2" >&2
    exit 1
fi

run_and_check() {
    local exe="$1" seed="$2" sites="$3" expected_n="$4" cap="$5"
    local R=8 name prefix
    name="1DIsing_N${expected_n}_R${R}_nSteps1_run${seed}"
    prefix="datasets/test/${name}"
    MCPA_DETAILED_CAP="${cap}" "${exe}" "${seed}" "${sites}" 1 "${R}" 1 \
        >"run-${seed}.log" 2>&1

    test -f "${prefix}_main.txt"
    test -f "${prefix}_agg_stats.txt"
    test -f "${prefix}_detailed_stats.txt"
    test "$(find datasets/test -maxdepth 1 -type f -name "${name}*" | wc -l)" -eq 3

    awk -F '\t' '
        NR == 1 {
            good = ($1 == "E" && $2 == "culling_factor" && $3 == "replica_family_avg_sq" && $4 == "nCull" && $6 == "status" && $8 == "equilibrate_seconds")
            if (!good) exit 1
            next
        }
        NR > 1 {
            if (($2 + 0.0) == 1.0) terminals++
            last = $2 + 0.0
            if (NR == 2) {
                for (i = 14; i <= 20; ++i) if ($i == "NA") exit 1
            } else {
                for (i = 14; i <= 20; ++i) if ($i != "NA") exit 1
            }
        }
        END { exit !(NR > 1 && terminals == 1 && last == 1.0) }
    ' "${prefix}_main.txt"
    awk -F '\t' 'NR == 1 { exit !($1 == "shell" && $4 == "accepted_flip_sum" && $7 == "M_sum" && $13 == "pre_family_count") }' \
        "${prefix}_agg_stats.txt"
    awk -F '\t' -v cap="${cap}" '
        FILENAME == ARGV[1] {
            if (FNR > 1) { shell = FNR - 2; main_e[shell] = $1; main_cull[shell] = $4 }
            next
        }
        FILENAME == ARGV[2] {
            if (FNR == 1) { if ($1 != "shell" || $3 != "shell_replica_count") bad = 1; next }
            shell = $1
            shells[shell] = 1
            agg_e[shell] = $2
            agg_count[shell] = $3
            if (main_e[shell] != $2 || main_cull[shell] != $3) bad = 1
            next
        }
        FILENAME == ARGV[3] {
            if (FNR == 1) {
                if ($9 != "total_matching_replicas" || $10 != "detailed_cap" || $11 != "sampling_policy") bad = 1
                next
            }
            shell = $1
            if (!(shell in shells) || $2 != agg_e[shell] || $3 < 0 || $3 >= 8) bad = 1
            if (seen[shell] && $3 <= last_replica[shell]) bad = 1
            if ($7 != (($6 < 0) ? -$6 : $6) || $8 != $6 * $6) bad = 1
            if ($9 != agg_count[shell] || $10 != cap || $11 != "first_matching_replica_indices") bad = 1
            seen[shell] = 1
            last_replica[shell] = $3
            rows[shell]++
        }
        END {
            for (shell in shells) {
                expected = (cap < 0 || agg_count[shell] < cap) ? agg_count[shell] : cap
                if (rows[shell] != expected) bad = 1
            }
            exit bad
        }
    ' "${prefix}_main.txt" "${prefix}_agg_stats.txt" "${prefix}_detailed_stats.txt"
}

run_and_check "${canonical}" 41 5 5 2
run_and_check "${legacy}" 42 3 9 2
run_and_check "${canonical}" 43 5 5 0
test "$(wc -l < datasets/test/1DIsing_N5_R8_nSteps1_run43_detailed_stats.txt)" -eq 1
run_and_check "${canonical}" 44 5 5 -1

for name in \
    1DIsing_N5_R8_nSteps1_run41 \
    1DIsing_N9_R8_nSteps1_run42 \
    1DIsing_N5_R8_nSteps1_run43 \
    1DIsing_N5_R8_nSteps1_run44; do
    for suffix in e.txt e2.txt X.txt pt.txt n.txt ch.txt time.txt; do
        if [[ -e "datasets/test/${name}${suffix}" ]]; then
            echo "FAIL: legacy output ${name}${suffix} was created" >&2
            exit 1
        fi
    done
done

echo "PASS: canonical/legacy CLIs emit exactly three versioned outputs with cap policy"
