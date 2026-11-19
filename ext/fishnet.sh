#!/bin/bash
# Months of Lichess/fishnet-evals (every position of analysed Lichess games with the server's Stockfish evaluation) -> shards.
# A separate line: these labels are Stockfish's. Kept: from move 9 on, side to move not in check, the move played not a
# capture or promotion, no forced mate.
#   bash ext/fishnet.sh <shard prefix> <shards> <months, e.g. 2025_02 2025_03 ...>
cd /root/loopchess; mkdir -p /root/ext; P=$1; K=$2; shift 2
say() { echo "[$(date -u +%H:%M:%S)] $*" >> fishnet.log; }
gcc -O2 -o ext/lconv ext/lconv.c -lm
say "months $*: download and conversion started"; touch zero/FARM_PAUSE
one() { m=$1; U="https://huggingface.co/datasets/Lichess/fishnet-evals/resolve/main/standard_rated_$m.parquet"
  size=$(curl -sIL "$U" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}'); parts=4; chunk=$(( (size + parts - 1) / parts ))
  for i in 0 1 2 3; do ( for try in 1 2 3 4; do curl -s -L --fail -r $(( i * chunk ))-$(( (i + 1) * chunk - 1 )) "$U" -o /root/ext/fn_$m.part$i && break; sleep 5; done ) & done; wait
  cat /root/ext/fn_$m.part0 /root/ext/fn_$m.part1 /root/ext/fn_$m.part2 /root/ext/fn_$m.part3 > /root/ext/fn_$m.parquet; rm -f /root/ext/fn_$m.part*
  python3 ext/pq2csv.py /root/ext/fn_$m.parquet fishnet | ext/lconv /root/ext/fn${P}_$m $K csv 9 2> /root/ext/fn${P}_$m.log; rm -f /root/ext/fn_$m.parquet
  echo "[$(date -u +%H:%M:%S)] $m: $(tail -1 /root/ext/fn${P}_$m.log)" >> fishnet.log; }
export -f one; export K P
printf '%s\n' "$@" | xargs -P 3 -I{} bash -c 'one {}'
rm -f zero/FARM_PAUSE
python3 - "$P" "$K" "$@" <<'PY'
import numpy as np, os, sys
P, K, months = sys.argv[1], int(sys.argv[2]), sys.argv[3:]; tot = 0
for k in range(K):
    c = np.concatenate([np.fromfile(f'/root/ext/fn{P}_{m}_{k}.c', dtype=np.uint8).reshape(-1, 24) for m in months]); y = np.concatenate([np.fromfile(f'/root/ext/fn{P}_{m}_{k}.y', dtype=np.uint8) for m in months])
    assert len(c) == len(y); p = np.random.RandomState(k).permutation(len(c)); np.save(f'data/ev_{P}{k}_c.npy', c[p]); np.save(f'data/ev_{P}{k}_y.npy', y[p]); tot += len(c)
    for m in months:
        os.remove(f'/root/ext/fn{P}_{m}_{k}.c'); os.remove(f'/root/ext/fn{P}_{m}_{k}.y')
open('fishnet.log', 'a').write(f'{tot/1e6:.0f}M positions kept in {K} shards data/ev_{P}<k>\n')
PY
say "$P finished"
