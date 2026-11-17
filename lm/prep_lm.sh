#!/bin/bash
# prep_lm.sh <first file> <last file>     the Lichess evaluation database (20 parquet files on Hugging Face) as shards
# data/lm_<i>_*.npy for the large transformer; one file at a time is downloaded, converted and deleted.
cd /root/loopchess; mkdir -p /root/ext
[ -s /root/ext/files.txt ] || curl -s "https://huggingface.co/api/datasets/Lichess/chess-position-evaluations/tree/main/data" | python3 -c "import json,sys; [print(i, x['path']) for i, x in enumerate(json.load(sys.stdin)) if x['path'].endswith('.parquet')]" > /root/ext/files.txt
for i in $(seq $1 $2); do
  [ -s data/lm_${i}_b.npy ] && continue
  path=$(awk -v i=$i '$1 == i {print $2}' /root/ext/files.txt)
  for try in 1 2 3; do curl -s -L --fail "https://huggingface.co/datasets/Lichess/chess-position-evaluations/resolve/main/$path" -o /root/ext/lm_$i.parquet && break; sleep 5; done
  echo "[$(date -u +%H:%M:%S)] $(python3 lm/prep_lm.py /root/ext/lm_$i.parquet data/lm_$i 22 2>&1 | tail -1)" >> lm_prep.log
  rm -f /root/ext/lm_$i.parquet
done
echo "[$(date -u +%H:%M:%S)] finished $1-$2" >> lm_prep.log
