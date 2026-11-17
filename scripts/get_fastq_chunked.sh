#!/usr/bin/env bash
# 分块顺序下载 + 每块校验大小，规避长连接被中断导致整文件重来
set -u
ACC=${1:-SRR30622183}
DIR=${2:-083}
OUT=${3:-data/fastq/ctrl_r1_SRR30622183.part.gz}
TOTAL=${4:-400000000}
CH=20000000
P="https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR306/${DIR}/${ACC}/${ACC}.fastq.gz"
: > "$OUT"
off=0
while [ $off -lt $TOTAL ]; do
  end=$((off+CH-1)); [ $end -ge $((TOTAL-1)) ] && end=$((TOTAL-1))
  want=$((end-off+1))
  got=0
  for try in 1 2 3 4 5 6; do
    curl -sS --max-time 180 -r $off-$end "$P" -o data/fastq/.chunk 2>/dev/null || true
    got=$(stat -c%s data/fastq/.chunk 2>/dev/null || echo 0)
    [ "$got" -eq "$want" ] && break
  done
  if [ "$got" -ne "$want" ]; then echo "FAIL at $off (got $got want $want)"; exit 1; fi
  cat data/fastq/.chunk >> "$OUT"
  off=$((off+CH))
  echo "$(date +%H:%M:%S) $off / $TOTAL"
done
rm -f data/fastq/.chunk
echo DONE $(stat -c%s "$OUT")
