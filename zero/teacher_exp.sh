#!/bin/bash
# Does a network trained on the two-brain teacher's targets play better than one trained on production targets?
#   tb  production target        tn  network + search, recalibrated to results (control)        to  the same + OURO
# Same positions, same steps; then to vs tn (what OURO adds), to vs tb, tn vs tb.
cd /root/loopchess; L=zero/teacher_exp.log; E=calc/calc_z5; ROWS=${ROWS:-20000000}
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
while [ ! -f data/ev_fs_pl.npy ] || [ ! -f zero/fss.npz ]; do sleep 10; done
sleep 30
OP=$(cat zero/ouro.pid 2>/dev/null); [ -n "$OP" ] && kill -STOP $OP && log "OURO's own training paused while it works as the teacher"
[ -f data/ev_fsT_yO.npy ] || python3 zero/teacher.py fs zero/g9.bin.ema.pt zero/ouro.pt $ROWS 2>&1 | grep -v Warning | tee -a $L
N=$(python3 -c "import numpy as np; print(len(np.load('data/ev_fsT_y.npy', mmap_mode='r')))")
EP=$(python3 -c "print(round(6000 * 16384 / $N, 4))")
train() {
  [ -f zero/fs_$1.bin.ema.pt ] && return
  python3 calc/train_nnue.py --h 256 --buckets 8 --ema 0.999 --files fsT --val fsval --epochs $EP --lr 1e-3 --threads 2 --device cuda --pst zero/fss.npz $2 --out zero/fs_$1.bin > zero/fs_$1.train.log 2>&1
  log "$1: $(grep '^done' zero/fs_$1.train.log | cut -c1-150)"
}
train to "--ytag yO" &
train tn "--ytag yN" &
wait
train tb "--ytag y"
[ -n "$OP" ] && kill -CONT $OP && log "OURO's training resumed"
for pair in "to tn" "to tb" "tn tb"; do
  set -- $pair
  grep -q "^$1 vs $2" zero/teacher_res.txt 2>/dev/null && continue
  r=$(python3 match.py --a calc:zero/fs_$1.bin:${TEST:-10000}:$E --b calc:zero/fs_$2.bin:${TEST:-10000}:$E --openings 1000 --workers 24 2>/dev/null | tail -1)
  echo "$1 vs $2: $r" >> zero/teacher_res.txt
  log "$1 vs $2: $(echo "$r" | python3 -c "import sys, json; d = json.loads(sys.stdin.read()); print(f\"{d['w']}-{d['d']}-{d['l']}  score {d['score']}  Elo {d['elo']} ({d['lo']} to {d['hi']})\")")"
done
log finished
