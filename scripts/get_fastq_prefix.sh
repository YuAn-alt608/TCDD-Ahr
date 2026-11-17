#!/usr/bin/env bash
# 受限带宽环境下只取每个文库的前 N 字节（gzip 流可按前缀解压）
# 用途：在真实 reads 上验证 TE 家族定量引擎；全量比对需完整 FASTQ（见 TE_count_readme.md）
set -u
N=${1:-45000000}
OUT=data/fastq
BASE=https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR306
run() { k=$1; p=$2; acc=${p##*/}
  echo "[$(date +%H:%M:%S)] $k $acc"
  curl -sS -L --retry 5 -r 0-$N "$BASE/$p/$acc.fastq.gz" -o "$OUT/${k}_${acc}.part.gz"
  echo "[$(date +%H:%M:%S)] $k $(du -h "$OUT/${k}_${acc}.part.gz" | cut -f1)"
}
run tcdd_r1 070/SRR30622170
run ctrl_r1 083/SRR30622183
echo PREFIX_DONE
