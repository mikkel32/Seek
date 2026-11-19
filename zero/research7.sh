#!/bin/bash
# Two questions, both decided by games:
#  (1) research6: does training on more games (ten shards instead of the four newest) make a stronger evaluator, and does
#      the wider evaluator need that extra data to pay off?
#  (2) Does showing positions mirrored left-right help? Chess is left-right symmetric once castling is gone, the evaluator
#      has no such symmetry built in, and research5 showed it is short of data, not of size: every mirrored position is a
#      free second example. M mirrors half of the rows whose kings have both left home, MA half of all rows, BM is M at 768 wide.
# Every arm starts from the same weights and trains for the same number of steps. The reference is A5 (384 wide, the four
# newest shards, no mirroring).
cd /root/loopchess; L=zero/research7.log; : > $L
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
while ! grep -q finished zero/research6.log; do sleep 10; done
recent=ar38,ar39,ar40,ar41; START=zero/r5_start.pt; PST=zero/g35s.npz; SAMPLES=362938368
arm() { n=$1; tags=$2; shift 2
  ep=$(python3 -c "import numpy as np,sys; print(round($SAMPLES / sum(len(np.load(f'data/ev_{t}_y.npy', mmap_mode='r')) for t in '$tags'.split(',')), 4))")
  t0=$(date +%s); python3 calc/train_nnue.py --buckets 8 --ema 0.999 --files $tags --val r6val --epochs $ep --lr 1e-3 --threads 4 --device cuda --pst $PST --widen $START "$@" --out zero/r7_$n.bin > zero/r7_$n.train.log 2>&1
  say "arm $n ($*): train loss $(grep -h '^step' zero/r7_$n.train.log | tail -1 | grep -o 'loss [0-9.]*') -> held-out $(grep -h '^done' zero/r7_$n.train.log | grep -o 'averaged ([0-9.]*' | cut -c11-18)  [$(( $(date +%s) - t0 ))s]"; }
arm M  $recent --h 384 --mirror 0.5
arm MA $recent --h 384 --mirror 0.5 --mirror_all 1
arm BM $recent --h 768 --mirror 0.5
cat > zero/r7_match.sh <<'EOM'
#!/bin/bash
cd /root/loopchess; L=zero/research7.log
for n in r6_A11 r6_B5 r6_B11 r7_M r7_MA r7_BM; do
  [ -s zero/$n.bin ] || { echo "$n has no network" >> $L; continue; }
  r=$(python3 gauntlet.py --us "calc/calc_w zero/$n.bin" --opp "A5|calc/calc_w zero/r6_A5.bin|Hash=64" --tc 10+0.1 --openings 150 --workers 48 --out zero/research7.jsonl | tail -1 | cut -c1-150)
  echo "[$(date -u +%H:%M:%S)] clock 10+0.1, $n vs A5: $r" >> $L
  r=$(python3 match.py --a calc:zero/$n.bin:20000:calc/calc_w --b calc:zero/r6_A5.bin:20000:calc/calc_w --openings 150 --workers 48 | tail -1 | cut -c1-110)
  echo "[$(date -u +%H:%M:%S)] 20,000 positions per move, $n vs A5: $r" >> $L
done
r=$(python3 gauntlet.py --us "calc/calc_w zero/r6_A5.bin" --opp "start|calc/calc_w zero/r5_S.bin|Hash=64" --tc 10+0.1 --openings 150 --workers 48 --out zero/research7.jsonl | tail -1 | cut -c1-150)
echo "[$(date -u +%H:%M:%S)] clock 10+0.1, A5 vs the start weights (generation 35): $r" >> $L
EOM
bash zero/burst.sh 48 bash zero/r7_match.sh
say "finished"
