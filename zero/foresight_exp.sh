#!/bin/bash
# Do training-only "foresight" targets or better value targets make a stronger network from the SAME games?
# Five networks from scratch on identical positions (the node-budget arms' games), then each plays the baseline.
#   base   production target (half search score, half result)
#   aux15  base + foresight heads, weight 0.15        aux50  the same, weight 0.5
#   next   half MY NEXT TURN's search score, half result
#   lam    lambda-return (0.95 per ply) over the game's later search scores and its result
cd /root/loopchess; L=zero/foresight_exp.log; E=calc/calc_z5
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
while ! grep -q "production restarted" zero/nodes_exp.log; do sleep 10; done
touch zero/KEEP_STOP; sleep 20
[ -f data/ev_fs_aux.npy ] || python3 zero/foresight.py fs "zero/raw/*.bin" 8 2>&1 | tee -a $L
[ -f zero/fss.npz ] || CUDA_VISIBLE_DEVICES= python3 calc/solve_pst.py 400000 fsp zero/fss > /dev/null 2>&1
N=$(python3 -c "import numpy as np; print(len(np.load('data/ev_fs_y.npy', mmap_mode='r')))")
EP=$(python3 -c "print(round(${STEPS:-6000} * 16384 / $N, 4))")
log "data: $N positions; every network trains ${STEPS:-6000} steps = $EP epochs, hidden ${HID:-256}"
train() {   # name, extra arguments
  [ -f zero/fs_$1.bin.ema.pt ] && return
  python3 calc/train_nnue.py --h ${HID:-256} --buckets 8 --ema 0.999 --files fs --val fsval --epochs $EP --lr 1e-3 --threads 2 --device cuda --pst zero/fss.npz $2 --out zero/fs_$1.bin > zero/fs_$1.train.log 2>&1
  log "$1: $(grep '^done' zero/fs_$1.train.log | cut -c1-150)  $(grep foresight zero/fs_$1.train.log | tail -1 | sed 's/.*foresight/| foresight/' | cut -c1-110)"
}
train base "" &
train aux15 "--aux 0.15" &
wait
train aux50 "--aux 0.5" &
train next "--ytag ys" &
wait
train lam "--ytag yl"
for v in aux15 aux50 next lam; do
  grep -q "^$v vs base" zero/foresight_res.txt 2>/dev/null && continue
  r=$(python3 match.py --a calc:zero/fs_$v.bin:${TEST:-10000}:$E --b calc:zero/fs_base.bin:${TEST:-10000}:$E --openings 1000 --workers 24 2>/dev/null | tail -1)
  echo "$v vs base: $r" >> zero/foresight_res.txt
  log "$v vs base: $(echo "$r" | python3 -c "import sys, json; d = json.loads(sys.stdin.read()); print(f\"{d['w']}-{d['d']}-{d['l']}  score {d['score']}  Elo {d['elo']} ({d['lo']} to {d['hi']})\")")"
done
log finished
