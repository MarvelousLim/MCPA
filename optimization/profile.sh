#!/usr/bin/env bash
set -euo pipefail

experiment="${1:-}"
profiler="${2:-ncu}"
preset="${3:-smoke}"
variant="${4:-candidate}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! "${experiment}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Usage: ./optimization/profile.sh EXPERIMENT [ncu|nsys] [PRESET] [VARIANT]" >&2
    exit 2
fi
export MCPA_PROFILE_DIR="${repo_dir}/optimization/artifacts/${experiment}"
mkdir -p "${MCPA_PROFILE_DIR}"
exec "${repo_dir}/profile.sh" "${profiler}" "${preset}" "${variant}"
