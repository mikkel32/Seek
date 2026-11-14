#!/bin/bash
# 8-minute, equal-wall-clock ablations on the small proxy config (d=96, 4 heads)
C="--d 96 --heads 4 --mlp 288 --minutes 8 --flat_lr --eval_every 150 --eval_n 2000 --total_steps 100000"
python3 train.py --name abl_full   $C                      > logs/abl_full.log 2>&1
python3 train.py --name abl_norule $C --no_rel --no_cnt    > logs/abl_norule.log 2>&1
python3 train.py --name abl_noloop $C --max_loops 1        > logs/abl_noloop.log 2>&1
python3 train.py --name abl_adamw  $C --no_muon --adam_lr 2e-3 > logs/abl_adamw.log 2>&1
echo ABLATIONS_DONE
