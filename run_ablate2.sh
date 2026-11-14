#!/bin/bash
# equal-wall-clock (10 min) ablations on the small proxy config (d=96, 4 heads, ~0.79M params)
C="--d 96 --heads 4 --mlp 288 --minutes 10 --flat_lr --eval_n 4000 --total_steps 100000"
python3 train.py --name ab2_full   $C --eval_every 200                 > logs/ab2_full.log 2>&1
python3 train.py --name ab2_noloop $C --eval_every 200 --max_loops 1   > logs/ab2_noloop.log 2>&1
python3 train.py --name ab2_bs128  $C --eval_every 400 --bs 128        > logs/ab2_bs128.log 2>&1
echo ABLATIONS2_DONE
