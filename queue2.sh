#!/bin/bash
# Pilot: does drawing half of the Lichess part from only-move positions beat plain sampling?
# Two arms from the same state (step 14,000), same learning rate (the floor), 1,000 steps each. Judged on the dev half only.
cd /home/user/loopchess; mkdir -p logs/done
run() { key=$1; shift; [ -f logs/done/$key ] && return; echo "[$(date -u +%H:%M:%S)] start $key"; bash -c "$*" && touch logs/done/$key; echo "[$(date -u +%H:%M:%S)] end $key"; }
ARCH="--d 160 --heads 5 --mlp 480 --n_in 2 --n_rec 3 --n_out 2 --max_loops 4 --bs 256 --warmup 200 --eval_every 500 --eval_n 3000"
run pilot_copy "cp ckpt/main_full_14000.pt ckpt/pilot_u.pt && cp ckpt/main_full_14000.pt ckpt/pilot_s.pt"
run pilot_u "python3 train.py --name pilot_u $ARCH --total_steps 15000 --sched_steps 12500 --minutes 30 >> logs/pilot_u.log 2>&1 && grep -q 'done 15000' logs/pilot_u.log"
run pilot_s "python3 train.py --name pilot_s $ARCH --total_steps 15000 --sched_steps 12500 --minutes 30 --sharp_frac 0.5 >> logs/pilot_s.log 2>&1 && grep -q 'done 15000' logs/pilot_s.log"
run pilot_eval "for n in B pilot_u pilot_s; do python3 bench.py ckpt/\$n.pt --half B --loops 2 --puzzles 0 > logs/dev_\$n.json 2>/dev/null; done"
echo queue2 done
