#!/usr/bin/env bash
# Rebuild every figure, Supplementary Table 1 and the report from processed_data/.
# No raw sequencing data required; needs numpy, pandas, scipy, matplotlib, openpyxl.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

for s in 09_figures.py 17_figures_story.py 24_figures_robust.py 28_figures_plan.py \
         34_figures_confound.py 36_figures_occupancy.py 44_figures_strengthen.py; do
  echo "== $s" && $PY "scripts/$s"
done

echo "== 45_table_s1.py" && $PY scripts/45_table_s1.py
echo "== 10_report.py"   && $PY scripts/10_report.py
echo "done: figures/, processed_data/Supplementary_Table_1_Claim_Audit.xlsx, report/"
