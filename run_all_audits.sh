#!/usr/bin/env bash
# Rebuild the audit chain behind Table S1: claim audit, external replication,
# donor-stratified branch statistics, stromal continuum, power floor.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
for s in 37_claim_audit.py 39_external_replication.py 41_human_branch_stats.py \
         42_stromal_continuum.py 43_power_floor.py 45_table_s1.py; do
  echo "== $s" && $PY "scripts/$s"
done
