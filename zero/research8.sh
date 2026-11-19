#!/bin/bash
# Two questions raised by research7, where an evaluator retrained on the newest games predicted unseen games BETTER and still
# lost 45 Elo to the weights it started from (generation 35).
#  (a) Is the production loop itself still going anywhere at clock speed? Current generation against generation 35 and
#      against generation 27 (the one the standing test used), and 35 against 27. 400 clock games each.
#  (b) What in the retraining does the damage: the number of passes, the size of the learning-rate restart, or the newer
#      games themselves? Four arms from generation 35's weights, each then plays those weights at the clock.
#        X1  four newest shards, one pass                X2  four newest shards, six passes, a tenth of the learning rate
#        X3  every kept shard, the usual learning rate   X4  every kept shard, a tenth of the learning rate
#      (X3 and X4 run the same number of steps as research6's arms.)
cd /root/loopchess; L=zero/research8.log; : > $L
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
while ! grep -q "^finished" zero/eval_gap.log 2>/dev/null; do sleep 10; done
G=$(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['gen'])"); cp zero/g$G.bin zero/r8_now.bin
recent=ar38,ar39,ar40,ar41; START=zero/r5_start.pt; PST=zero/g35s.npz; SAMPLES=362938368
all="r5s26,r5s27,r5s28,r5s29,r5s30,r5s31"; for f in data/ev_ar*_x.npy; do k=${f#data/ev_ar}; k=${k%_x.npy}; [ "$k" != 42 ] && all="$all,ar$k"; done
say "start: current generation $G; every kept shard = $all (shard 42 stays held out)"
arm() { n=$1; tags=$2; smp=$3; shift 3
  ep=$(python3 -c "import numpy as np,sys; print(round($smp / sum(len(np.load(f'data/ev_{t}_y.npy', mmap_mode='r')) for t in '$tags'.split(',')), 4))")
  t0=$(date +%s); python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files $tags --val r6val --epochs $ep --threads 4 --device cuda --pst $PST --widen $START "$@" --out zero/r8_$n.bin > zero/r8_$n.train.log 2>&1
  say "arm $n ($* ; $ep passes over $(echo $tags | tr ',' '\n' | wc -l) shards): train loss $(grep -h '^step' zero/r8_$n.train.log | tail -1 | grep -o 'loss [0-9.]*') -> held-out $(grep -h '^done' zero/r8_$n.train.log | grep -o 'averaged ([0-9.]*' | cut -c11-18)  [$(( $(date +%s) - t0 ))s]"; }
arm X1 $recent 61300000   --lr 1e-3
arm X2 $recent $SAMPLES   --lr 1e-4
arm X3 $all    $SAMPLES   --lr 1e-3
arm X4 $all    $SAMPLES   --lr 1e-4
cat > zero/r8_match.sh <<'EOM'
#!/bin/bash
cd /root/loopchess; L=zero/research8.log
g() { r=$(python3 gauntlet.py --us "$1 $2" --opp "$4|$1 $3|Hash=64" --tc 10+0.1 --openings $5 --workers 48 --out zero/research8.jsonl | tail -1 | cut -c1-150); echo "[$(date -u +%H:%M:%S)] clock 10+0.1, $6: $r" >> $L; }
g calc/calc_untuned zero/r8_now.bin zero/r5_S.bin "generation 35" 200 "current generation vs generation 35"
g calc/calc_untuned zero/r5_S.bin zero/test_net.bin "generation 27" 200 "generation 35 vs generation 27"
g calc/calc_untuned zero/r8_now.bin zero/test_net.bin "generation 27" 200 "current generation vs generation 27"
for n in X1 X2 X3 X4; do
  [ -s zero/r8_$n.bin ] && g calc/calc_w zero/r8_$n.bin zero/r5_S.bin "generation 35" 150 "$n vs generation 35 (its start)"
done
EOM
bash zero/burst.sh 48 bash zero/r8_match.sh
say "finished"
