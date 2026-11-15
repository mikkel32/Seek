#!/bin/bash
# The experiments the old machine did not finish, on the new one. Sized to stay under the CPU allowance beside production:
# matches use 8 workers, GPU jobs get priority.
#   1. value targets: networks trained on "next turn's score" and on the lambda-return, each against the baseline
#   2. two-brain teacher: tb production target, tn network+search recalibrated (control), to the same + OURO
cd /root/loopchess; L=zero/research2.log; E=calc/calc_z5
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
match() {
  grep -q "^$1 vs $2:" zero/research2_res.txt 2>/dev/null && return
  r=$(python3 match.py --a calc:zero/fs_$1.bin:10000:$E --b calc:zero/fs_$2.bin:10000:$E --openings 1000 --workers 8 2>/dev/null | tail -1)
  echo "$1 vs $2: $r" >> zero/research2_res.txt
  log "$1 vs $2: $(echo "$r" | python3 -c "import sys, json; d = json.loads(sys.stdin.read()); print(f\"{d['w']}-{d['d']}-{d['l']}  score {d['score']}  Elo {d['elo']} ({d['lo']} to {d['hi']})\")")"
}
( match next base; match lam base ) &
[ -f data/ev_fsST_yO.npy ] || nice -n -10 python3 zero/teacher.py fsS zero/g9.bin.ema.pt zero/ouro_snap.pt 20000000 2>&1 | grep -v Warning | tee -a $L
N=$(python3 -c "import numpy as np; print(len(np.load('data/ev_fsST_y.npy', mmap_mode='r')))")
EP=$(python3 -c "print(round(6000 * 16384 / $N, 4))")
train() {
  [ -f zero/fs_$1.bin.ema.pt ] && return
  nice -n -10 python3 calc/train_nnue.py --h 256 --buckets 8 --ema 0.999 --files fsST --val fsval --epochs $EP --lr 1e-3 --threads 2 --device cuda --pst zero/fss.npz $2 --out zero/fs_$1.bin > zero/fs_$1.train.log 2>&1
  log "$1: $(grep '^done' zero/fs_$1.train.log | cut -c1-150)"
}
train to "--ytag yO"; train tn "--ytag yN"; train tb "--ytag y"
wait
match to tn; match to tb; match tn tb
log finished
