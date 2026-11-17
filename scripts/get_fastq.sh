#!/usr/bin/env bash
# 断点续传下载 D2 scRNA (tdT+ = SMC 分选) Ctrl/TCDD 各 2 个重复
set -u
OUT=data/fastq
BASE=https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR306
declare -A RUNS=(
  [ctrl_r1]=083/SRR30622183
  [ctrl_r2]=086/SRR30622186
  [tcdd_r1]=070/SRR30622170
  [tcdd_r2]=069/SRR30622169
)
for k in tcdd_r1 ctrl_r1 tcdd_r2 ctrl_r2; do
  p=${RUNS[$k]}; acc=${p##*/}
  echo "[$(date +%H:%M:%S)] start $k $acc"
  curl -sS -L --retry 8 --retry-delay 5 -C - "$BASE/$p/$acc.fastq.gz" -o "$OUT/${k}_${acc}.fastq.gz"
  echo "[$(date +%H:%M:%S)] done  $k $(du -h "$OUT/${k}_${acc}.fastq.gz" | cut -f1)"
done
echo "ALL_DONE"
