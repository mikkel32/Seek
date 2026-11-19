#!/bin/bash
# Is the engine short of data rather than of network size?
# research5 found that bigger evaluators fit their training window better and unseen positions worse. If the window is
# the limit, the same evaluator trained on more games must do better on unseen positions, and the bigger one must gain more.
# Four arms, the same start weights, the same number of training steps; only the amount of data and the width differ.
# Held-out positions come from a newer shard (r6val, made by zero/research6a.sh) that no arm trains on.
#   recent  = the five shards just before the held-out one          both = those plus six older shards (r5s26..31)
cd /root/loopchess; L=zero/research6.log; : > $L
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
VK=${VK:-42}; START=zero/r5_start.pt; PST=zero/g35s.npz; SAMPLES=362938368
recent=""; for i in $(seq $((VK - 5)) $((VK - 1))); do
  [ -e data/ev_ar${i}_x.npy ] || { ln -f data/ev_sh${i}_x.npy data/ev_ar${i}_x.npy 2>/dev/null && ln -f data/ev_sh${i}_y.npy data/ev_ar${i}_y.npy 2>/dev/null; }
  [ -e data/ev_ar${i}_y.npy ] && recent="$recent,ar$i"; done; recent=${recent#,}
old="r5s26,r5s27,r5s28,r5s29,r5s30,r5s31"; both="$old,$recent"
say "start: recent = $recent; both = $both; held-out shard $VK"
arm() { n=$1; tags=$2; shift 2
  ep=$(python3 -c "import numpy as np,sys; print(round($SAMPLES / sum(len(np.load(f'data/ev_{t}_y.npy', mmap_mode='r')) for t in '$tags'.split(',')), 4))")
  t0=$(date +%s); python3 calc/train_nnue.py --buckets 8 --ema 0.999 --files $tags --val r6val --epochs $ep --lr 1e-3 --threads 4 --device cuda --pst $PST --widen $START "$@" --out zero/r6_$n.bin > zero/r6_$n.train.log 2>&1
  say "arm $n ($* on $(echo $tags | tr ',' '\n' | wc -l) shards, $ep epochs): $(grep -h '^positions' zero/r6_$n.train.log | cut -c1-34) train loss $(grep -h '^step' zero/r6_$n.train.log | tail -1 | grep -o 'loss [0-9.]*') -> held-out $(grep -h '^done' zero/r6_$n.train.log | grep -o 'averaged ([0-9.]*' | cut -c11-18)  [$(( $(date +%s) - t0 ))s]"; }
arm A5  $recent --h 384
arm A11 $both   --h 384
arm B5  $recent --h 768
arm B11 $both   --h 768
say "finished"
