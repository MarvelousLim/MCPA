#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point.  Test and benchmark builds now share test.sh;
# optimization scripts can keep calling benchmark.sh unchanged.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${repo_dir}/test.sh" benchmark "$@"
