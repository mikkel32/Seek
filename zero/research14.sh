#!/bin/bash
# Two questions on the full long-memory window, each arm a normal generation step (240M samples from the same start):
#  1. How much do two trainings of the SAME recipe differ in play, when only the random seed changes? Every "+20 Elo"
#     in these notes has to be read against that number, and it has never been measured.
#  2. "Beyond the search": research12's control label (search score pushed 12% further away from the evaluator's own
#     opinion, the weights a fit to game results chose) beat production's label by 20 Elo on a small window. Here the
#     same push is applied to production's labels on the whole window (calc/train_nnue.py --extrap), at two doses.
#   E0   the loop's recipe, seed 0 (control)        E0b  the same, seed 1
#   X12  labels pushed 12% beyond                   X30  labels pushed 30% beyond
# Judged at 20,000 positions per move (2,000 games per pairing) and at the clock (300 games each).
cd /root/loopchess; L=zero/research14.log; : > $L
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
while ! grep -q "\] finished" zero/research13.log 2>/dev/null; do sleep 20; done
G=$(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['gen'])")
tags=$(python3 -c "import json;print(','.join('sh%d' % i for i in json.load(open('zero/cont_state.json'))['shards'][1:]))")     # the oldest shard may leave the window while this runs
START=zero/r14_start.pt; cp zero/g$G.bin.ema.pt $START; PST=$(ls zero/g${G}s.npz zero/g$((G - 1))s.npz 2>/dev/null | head -1)
cp data/ev_shval_x.npy data/ev_r14val_x.npy; cp data/ev_shval_y.npy data/ev_r14val_y.npy
for t in $(echo $tags | tr ',' ' '); do ln -f data/ev_${t}_x.npy data/ev_k${t}_x.npy 2>/dev/null && ln -f data/ev_${t}_y.npy data/ev_k${t}_y.npy 2>/dev/null; done      # frozen names: the loop may drop shards meanwhile
tags=$(for t in $(echo $tags | tr ',' ' '); do [ -e data/ev_k${t}_y.npy ] && printf "k$t,"; done | sed 's/,$//')
say "start: generation $G weights, $(echo $tags | tr ',' '\n' | wc -l) shards"
ep=$(python3 -c "import numpy as np; print(round(240000000 / sum(len(np.load(f'data/ev_{t}_y.npy', mmap_mode='r')) for t in '$tags'.split(',')), 4))")
arm() { n=$1; shift
  t0=$(date +%s); python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files $tags --val r14val --epochs $ep --lr 1e-3 --threads 4 --device cuda --pst $PST --widen $START "$@" --out zero/r14_$n.bin > zero/r14_$n.train.log 2>&1
  say "arm $n ($* ; $ep passes): played held-out $(grep -h '^done' zero/r14_$n.train.log | grep -o 'averaged ([0-9.]*' | cut -c11-18), opening position $(printf 'position startpos\neval\nquit\n' | calc/calc_w zero/r14_$n.bin | grep '^eval')  [$(( $(date +%s) - t0 ))s]"; }
arm E0 --seed 0
arm E0b --seed 1
arm X12 --seed 0 --extrap 0.12
arm X30 --seed 0 --extrap 0.30
rm -f data/ev_ksh*_[xy].npy
cat > zero/r14_match.sh <<'EOM'
#!/bin/bash
cd /root/loopchess; L=zero/research14.log
m() { for try in 1 2; do r=$(python3 match.py --a calc:zero/r14_$1.bin:20000:calc/calc_w --b calc:zero/r14_$2.bin:20000:calc/calc_w --openings 1000 --workers 48 | tail -1 | cut -c1-110); [ -n "$r" ] && break; sleep 5; done
      echo "[$(date -u +%H:%M:%S)] 20,000 positions per move, $1 vs $2: $r" >> $L; }
c() { r=$(python3 gauntlet.py --us "calc/calc_w zero/r14_$1.bin" --opp "$2|calc/calc_w zero/r14_$2.bin|Hash=64" --tc 10+0.1 --openings 150 --workers 48 --out zero/research14.jsonl | tail -1 | cut -c1-150)
      echo "[$(date -u +%H:%M:%S)] clock 10+0.1, $1 vs $2: $r" >> $L; }
sleep 10
m E0b E0; m X12 E0; m X30 E0
c X12 E0; c X30 E0; c E0b E0
EOM
bash zero/burst.sh 48 bash zero/r14_match.sh
say "finished"
