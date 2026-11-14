#!/bin/bash
# Phase specialists trained to agree at their boundaries (B2), against the wide network (A) and the first specialists (B).
cd "$(dirname "$0")/.."
D="--buckets 8 --ema 0.999 --files z3t,z4t --val z4v --epochs 2 --lr 1e-3 --pst zero/g5s.npz --seed 3 --threads 4"
echo "validation log-loss B2 (8 specialists x 64, boundary overlap 0.3): $(python3 calc/train_nnue.py --h 64 --kb 8 --phase --overlap 0.3 $D --out zero/pb2.bin 2>&1 | grep -E '^done' | sed 's/.*averaged (\([0-9.]*\), \([0-9.]*\)).*/\1/')"
python3 calc/check_net.py zero/pb2.bin.ema.pt zero/pb2.bin calc/calc_x
python3 match.py --a calc:zero/pb2.bin:0.05:calc/calc_x --b calc:zero/pa.bin:0.05:calc/calc_x --openings 300 --workers 4 2>/dev/null | tail -1 | python3 -c "import sys, json; d = json.loads(sys.stdin.read()); print('B2 vs A, equal time:', d['w'], d['d'], d['l'], 'score', d['score'], 'Elo', d['elo'], (d['lo'], d['hi']))"
