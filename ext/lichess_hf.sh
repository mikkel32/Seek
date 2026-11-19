#!/bin/bash
# The Lichess evaluation database (Hugging Face copy, 20 parquet files, labels by Stockfish) as training data -- a separate
# line, NOT the self-taught one. Seven files at a time are downloaded and converted (ext/pq2csv.py | ext/lconv); the
# self-play farm is paused meanwhile. Then shards data/ev_e<k> and a plain 384 network from scratch on them.
cd /root/loopchess; mkdir -p /root/ext; K=${1:-12}; P=${2:-8}
say() { echo "[$(date -u +%H:%M:%S)] $*" >> lichess.log; }
gcc -O2 -o ext/lconv ext/lconv.c -lm
curl -s "https://huggingface.co/api/datasets/Lichess/chess-position-evaluations/tree/main/data" | python3 -c "import json,sys; [print(i, x['path']) for i, x in enumerate(json.load(sys.stdin)) if x['path'].endswith('.parquet')]" > /root/ext/files.txt
say "download and conversion of $(wc -l < /root/ext/files.txt) files started"; touch zero/FARM_PAUSE
one() { i=$1; path=$2; for try in 1 2 3; do curl -s -L --fail "https://huggingface.co/datasets/Lichess/chess-position-evaluations/resolve/main/$path" -o /root/ext/p$i.parquet && break; sleep 5; done
  python3 ext/pq2csv.py /root/ext/p$i.parquet | ext/lconv /root/ext/le${i} $K csv 2> /root/ext/lconv_$i.log; rm -f /root/ext/p$i.parquet; }
export -f one; export K
xargs -P 7 -L 1 bash -c 'one $0 $1' < /root/ext/files.txt
rm -f zero/FARM_PAUSE
python3 - <<PY
import numpy as np, os, glob, re
n = len(open('/root/ext/files.txt').read().split('\n')) - 1; tot = 0; read = kept = 0
for f in glob.glob('/root/ext/lconv_*.log'):
    m = re.search(r'(\d+) positions read; kept (\d+)', open(f).read()); read += int(m.group(1)); kept += int(m.group(2))
for k in range($K):
    c = np.concatenate([np.fromfile(f'/root/ext/le{i}_{k}.c', dtype=np.uint8).reshape(-1, 24) for i in range(n)]); y = np.concatenate([np.fromfile(f'/root/ext/le{i}_{k}.y', dtype=np.uint8) for i in range(n)])
    assert len(c) == len(y); p = np.random.RandomState(k).permutation(len(c)); np.save(f'data/ev_e{k}_c.npy', c[p]); np.save(f'data/ev_e{k}_y.npy', y[p]); tot += len(c)
    for i in range(n):
        os.remove(f'/root/ext/le{i}_{k}.c'); os.remove(f'/root/ext/le{i}_{k}.y')
open('lichess.log', 'a').write(f'{read/1e6:.1f}M positions in the database, {kept/1e6:.1f}M kept (quiet, not in check, no forced mate) in {$K} shards\n')
PY
tags=$(seq -s, -f "e%g" 0 $((K-1)))
say "training a plain 384 network from scratch on them, $P passes"
t0=$(date +%s); python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files $tags --val bval --epochs $P --lr 1e-3 --threads 4 --device cuda --pst zero/g101s.npz --out zero/w_LE.bin > zero/w_LE.train.log 2>&1
say "network trained on Stockfish-labelled Lichess positions (zero/w_LE.bin) in $(( $(date +%s) - t0 ))s: $(grep -h '^done' zero/w_LE.train.log | cut -c1-140)"
say finished
