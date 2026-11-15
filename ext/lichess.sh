#!/bin/bash
# The Lichess evaluation database as training data (labels by Stockfish -- a separate line, NOT the self-taught one):
# download and convert in one stream, make shards data/ev_e<k>, train a plain 384 network from scratch on them.
cd /root/loopchess; mkdir -p /root/ext; K=${1:-24}; P=${2:-8}
say() { echo "[$(date -u +%H:%M:%S)] $*" >> lichess.log; }
gcc -O2 -o ext/lconv ext/lconv.c -lm
say "download and conversion started"
curl -s https://database.lichess.org/lichess_db_eval.jsonl.zst | zstd -dc --long=31 | ext/lconv /root/ext/le $K 2> /root/ext/lconv.log
say "$(tail -1 /root/ext/lconv.log)"
python3 - <<PY
import numpy as np, os
tot = 0
for k in range($K):
    c = np.fromfile(f'/root/ext/le_{k}.c', dtype=np.uint8).reshape(-1, 24); y = np.fromfile(f'/root/ext/le_{k}.y', dtype=np.uint8)
    assert len(c) == len(y); np.save(f'data/ev_e{k}_c.npy', c); np.save(f'data/ev_e{k}_y.npy', y); tot += len(c)
    os.remove(f'/root/ext/le_{k}.c'); os.remove(f'/root/ext/le_{k}.y')
print(tot)
PY
tags=$(seq -s, -f "e%g" 0 $((K-1)))
say "$K shards written; training a plain 384 network from scratch, $P passes"
t0=$(date +%s); python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files $tags --val bval --epochs $P --lr 1e-3 --threads 4 --device cuda --pst zero/g101s.npz --out zero/w_LE.bin > zero/w_LE.train.log 2>&1
say "network trained on Stockfish-labelled Lichess positions (zero/w_LE.bin) in $(( $(date +%s) - t0 ))s: $(grep -h '^done' zero/w_LE.train.log | cut -c1-140)"
say finished
