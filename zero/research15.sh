#!/bin/bash
# Product terms in the fast evaluator. Today its output is a weighted sum of SQUARES of hidden units (each unit a sum over
# pieces). A square can express pairs of pieces only through one unit; a PRODUCT of two different units is an "and" of
# two learned patterns: two of my own, or one of mine and one of the opponent's. The transformer, trained on the same
# games, judges positions about as well as a 2,500-position search of this evaluator, so the evaluator's form, not only
# its data, holds it back; products add piece-pair interactions with 6,000 extra output weights and no change to the
# big input table (which is where width and king regions overfitted on a short memory).
#   P0   the loop's evaluator, 720M samples from the current generation
#   P1   the same plus product terms (starting at zero: the same function at step 0), 720M samples
# Judged at 20,000 positions per move (2,000 games) and at the clock (600 games), both arms in the same engine build.
cd /root/loopchess; L=zero/research15.log; : > $L
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
G=$(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['gen'])")
tags=$(python3 -c "import json;print(','.join('sh%d' % i for i in json.load(open('zero/cont_state.json'))['shards'][2:]))")     # the two oldest shards may leave the window while this runs
START=zero/r15_start.pt; cp zero/g$G.bin.ema.pt $START; PST=$(ls zero/g${G}s.npz zero/g$((G - 1))s.npz 2>/dev/null | head -1)
cp data/ev_shval_x.npy data/ev_r15val_x.npy; cp data/ev_shval_y.npy data/ev_r15val_y.npy
for t in $(echo $tags | tr ',' ' '); do ln -f data/ev_${t}_x.npy data/ev_p${t}_x.npy 2>/dev/null && ln -f data/ev_${t}_y.npy data/ev_p${t}_y.npy 2>/dev/null; done      # frozen names: the loop may drop shards meanwhile
tags=$(for t in $(echo $tags | tr ',' ' '); do [ -e data/ev_p${t}_y.npy ] && printf "p$t,"; done | sed 's/,$//')
say "start: generation $G weights, $(echo $tags | tr ',' '\n' | wc -l) shards"
arm() { n=$1; smp=$2; shift 2
  ep=$(python3 -c "import numpy as np; print(round($smp / sum(len(np.load(f'data/ev_{t}_y.npy', mmap_mode='r')) for t in '$tags'.split(',')), 4))")
  t0=$(date +%s); python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files $tags --val r15val --epochs $ep --lr 1e-3 --threads 4 --device cuda --pst $PST --widen $START "$@" --out zero/r15_$n.bin > zero/r15_$n.train.log 2>&1
  say "arm $n ($* ; $ep passes): train loss $(grep -h '^step' zero/r15_$n.train.log | tail -1 | grep -o 'loss [0-9.]*') -> played held-out $(grep -h '^done' zero/r15_$n.train.log | grep -o 'averaged ([0-9.]*' | cut -c11-18)  [$(( $(date +%s) - t0 ))s]"; }
arm P0 ${SMP:-720000000}
arm P1 ${SMP:-720000000} --prod 1
rm -f data/ev_psh*_[xy].npy
say "engine against trainer, P0: $(python3 zero/check_net.py calc/calc_p zero/r15_P0.bin zero/r15_P0.bin.ema.pt 2>&1 | tail -1)"
say "engine against trainer, P1: $(python3 zero/check_net.py calc/calc_p zero/r15_P1.bin zero/r15_P1.bin.ema.pt 2>&1 | tail -1)"
say "speed alone, positions per second: P0 $(printf 'position startpos\ngo movetime 3000\nquit\n' | calc/calc_p zero/r15_P0.bin | grep -o 'nps [0-9]*' | tail -1)   P1 $(printf 'position startpos\ngo movetime 3000\nquit\n' | calc/calc_p zero/r15_P1.bin | grep -o 'nps [0-9]*' | tail -1)"
while ! grep -q "\] finished" zero/research14.log 2>/dev/null; do sleep 20; done
cat > zero/r15_match.sh <<'EOM'
#!/bin/bash
cd /root/loopchess; L=zero/research15.log
sleep 10
for try in 1 2; do r=$(python3 match.py --a calc:zero/r15_P1.bin:20000:calc/calc_p --b calc:zero/r15_P0.bin:20000:calc/calc_p --openings 1000 --workers 48 | tail -1 | cut -c1-110); [ -n "$r" ] && break; sleep 5; done
echo "[$(date -u +%H:%M:%S)] 20,000 positions per move, P1 vs P0: $r" >> $L
r=$(python3 gauntlet.py --us "calc/calc_p zero/r15_P1.bin" --opp "P0|calc/calc_p zero/r15_P0.bin|Hash=64" --tc 10+0.1 --openings 300 --workers 48 --out zero/research15.jsonl | tail -1 | cut -c1-150)
echo "[$(date -u +%H:%M:%S)] clock 10+0.1, P1 vs P0: $r" >> $L
EOM
bash zero/burst.sh 48 bash zero/r15_match.sh
say "finished"
