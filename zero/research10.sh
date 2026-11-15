#!/bin/bash
# Does a wider evaluator pay off now that the loop remembers 27+ shards (about 400M positions)?
# research5-7: on 4 to 6 shards 768 wide was worse per position searched; on 10 shards it drew level. Here three arms start
# from the current production weights and train on the whole window:
#   W384   384 wide, 240M samples (what the loop's next generation does: the control)
#   W768   768 wide, 480M samples          W768L  768 wide, 960M samples (the new units need time to learn)
# Judged at a fixed 20,000 positions per move (is it smarter?) and at the clock (does that survive being slower?).
cd /root/loopchess; L=zero/research10.log; : > $L
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
G=$(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['gen'])")
tags=$(python3 -c "import json;print(','.join('sh%d' % i for i in json.load(open('zero/cont_state.json'))['shards']))")
START=zero/r10_start.pt; cp zero/g$G.bin.ema.pt $START; cp zero/g$G.bin zero/r10_S.bin; PST=$(ls zero/g${G}s.npz zero/g$((G - 1))s.npz 2>/dev/null | head -1)
cp data/ev_shval_x.npy data/ev_r10val_x.npy; cp data/ev_shval_y.npy data/ev_r10val_y.npy
say "start: generation $G weights, $(echo $tags | tr ',' '\n' | wc -l) shards"
arm() { n=$1; smp=$2; shift 2
  ep=$(python3 -c "import numpy as np; print(round($smp / sum(len(np.load(f'data/ev_{t}_y.npy', mmap_mode='r')) for t in '$tags'.split(',')), 4))")
  t0=$(date +%s); python3 calc/train_nnue.py --buckets 8 --ema 0.999 --files $tags --val r10val --epochs $ep --lr 1e-3 --threads 4 --device cuda --pst $PST --widen $START "$@" --out zero/r10_$n.bin > zero/r10_$n.train.log 2>&1
  say "arm $n ($* ; $ep passes): train loss $(grep -h '^step' zero/r10_$n.train.log | tail -1 | grep -o 'loss [0-9.]*') -> played held-out $(grep -h '^done' zero/r10_$n.train.log | grep -o 'averaged ([0-9.]*' | cut -c11-18)  [$(( $(date +%s) - t0 ))s]"; }
arm W384  240000000 --h 384
arm W768  480000000 --h 768
arm W768L 960000000 --h 768
cat > zero/r10_match.sh <<'EOM'
#!/bin/bash
cd /root/loopchess; L=zero/research10.log
for n in W768 W768L; do
  [ -s zero/r10_$n.bin ] || continue
  r=$(python3 match.py --a calc:zero/r10_$n.bin:20000:calc/calc_w --b calc:zero/r10_W384.bin:20000:calc/calc_w --openings 200 --workers 48 | tail -1 | cut -c1-110)
  echo "[$(date -u +%H:%M:%S)] 20,000 positions per move, $n vs W384: $r" >> $L
  r=$(python3 gauntlet.py --us "calc/calc_w zero/r10_$n.bin" --opp "W384|calc/calc_w zero/r10_W384.bin|Hash=64" --tc 10+0.1 --openings 150 --workers 48 --out zero/research10.jsonl | tail -1 | cut -c1-150)
  echo "[$(date -u +%H:%M:%S)] clock 10+0.1, $n vs W384: $r" >> $L
done
r=$(python3 gauntlet.py --us "calc/calc_w zero/r10_W384.bin" --opp "start|calc/calc_w zero/r10_S.bin|Hash=64" --tc 10+0.1 --openings 150 --workers 48 --out zero/research10.jsonl | tail -1 | cut -c1-150)
echo "[$(date -u +%H:%M:%S)] clock 10+0.1, W384 vs its start: $r" >> $L
EOM
bash zero/burst.sh 48 bash zero/r10_match.sh
say "finished"
