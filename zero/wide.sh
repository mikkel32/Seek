#!/bin/bash
# A wider evaluator on fresh positions (zero/farm.py), trained and judged on one machine. Two questions:
#  width:   does twice the width pay on this much fresh data?
#  recipe:  other self-taught engines train each network for dozens of passes with the learning rate decayed to 1e-6
#           (the loop trains half a pass per generation and restarts the learning rate every time). Does a long run pay?
#   N384   current width, 2 passes            N384L  current width, 16 passes, learning rate down to 1e-6
#   W768   twice the width, 3 passes          W768L  twice the width, 16 passes, learning rate down to 1e-6
# All start from generation 101 (new units start with zero output weights: the same function at step 0). Judged with the
# integer engine built for wide networks: at 30,000 positions per move, then at the clock (10+0.1 and 60+0.6).
cd /root/loopchess; L=wide.log; : > $L
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
until [ "$(ls data/ev_b[0-9]*_c.npy 2>/dev/null | wc -l)" -ge ${NSH:-36} ] || [ -e zero/WIDE_GO ]; do sleep 30; done
sleep 20
tags=$(ls data/ev_b[0-9]*_c.npy | sed 's/.*ev_\(b[0-9]*\)_c.npy/\1/' | sort -V | tr '\n' ',' | sed 's/,$//')
say "$(echo $tags | tr ',' '\n' | wc -l) shards, $(python3 -c "import numpy as np; print(round(sum(len(np.load(f'data/ev_{t}_y.npy', mmap_mode='r')) for t in '$tags'.split(',')) / 1e6))")M fresh positions"
arm() { n=$1; h=$2; ep=$3; shift 3; t0=$(date +%s)
  python3 calc/train_nnue.py --h $h --buckets 8 --ema 0.999 --files $tags --val bval --epochs $ep --lr 1e-3 --threads 4 --device cuda --pst zero/g101s.npz --widen zero/g101.pt "$@" --out zero/w_$n.bin > zero/w_$n.train.log 2>&1
  say "arm $n (width $h, $ep passes $*): played held-out $(grep -h '^done' zero/w_$n.train.log | grep -o 'averaged ([0-9.]*' | cut -c11-18) (start $(grep -h '^positions' zero/w_$n.train.log | grep -o 'MAE) ([0-9.]*' | cut -c7-14)), opening position $(printf 'position startpos\neval\nquit\n' | calc/calc_s10w zero/w_$n.bin | grep '^eval')  [$(( $(date +%s) - t0 ))s]"; }
g() { bash gt.sh "$1" wide.log 76 --us "calc/calc_s10w zero/w_$2.bin" --opp "$3|calc/calc_s10w $4|Hash=64" "${@:5}"; }
arm N384 384 2
arm W768 768 3
( arm W768L 768 16 --lr_end 1e-6; arm N384L 384 16 --lr_end 1e-6 ) &
for n in N384 W768; do say "engine against trainer, $n: $(python3 zero/check_net.py calc/calc_s10w zero/w_$n.bin zero/w_$n.bin.ema.pt 2>&1 | tail -1 | cut -c1-150)"; done
for n in N384 W768; do say "speed alone, $n: $(python3 zero/bench.py calc/calc_s10w zero/w_$n.bin 14 | cut -c1-90)"; done
g "fresh data alone: N384 against generation 101, 30,000 positions per move" N384 "generation 101" zero/vol_net.bin --openings 750 --nodes 30000 --stop=-10,10
g "W768 against N384, 30,000 positions per move" W768 N384 zero/w_N384.bin --openings 1000 --nodes 30000 --stop 0,20
g "W768 against N384 at the clock, 10+0.1" W768 N384 zero/w_N384.bin --tc 10+0.1 --openings 300 --stop 0,20
g "W768 against generation 101 at the clock, 10+0.1" W768 "generation 101" zero/vol_net.bin --tc 10+0.1 --openings 300 --stop 0,30
g "W768 against generation 101 at the clock, 60+0.6" W768 "generation 101" zero/vol_net.bin --tc 60+0.6 --openings 150 --stop 0,30
wait
g "long training: W768L against W768, 30,000 positions per move" W768L W768 zero/w_W768.bin --openings 1000 --nodes 30000 --stop 0,20
g "long training: N384L against N384, 30,000 positions per move" N384L N384 zero/w_N384.bin --openings 1000 --nodes 30000 --stop 0,20
g "W768L against generation 101 at the clock, 10+0.1" W768L "generation 101" zero/vol_net.bin --tc 10+0.1 --openings 300 --stop 0,30
g "W768L against generation 101 at the clock, 60+0.6" W768L "generation 101" zero/vol_net.bin --tc 60+0.6 --openings 150 --stop 0,30
g "N384L against generation 101 at the clock, 10+0.1" N384L "generation 101" zero/vol_net.bin --tc 10+0.1 --openings 300 --stop 0,30
g "N384L against generation 101 at the clock, 60+0.6" N384L "generation 101" zero/vol_net.bin --tc 60+0.6 --openings 150 --stop 0,30
say "finished"
