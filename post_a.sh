#!/bin/bash
# Stage 1 after training: freeze checkpoint A, run the weight-space-programming recipe, train the new checkpoint B.
set -e
cd /home/user/loopchess
python3 make_ckpt.py ckpt/main.pt ckpt/A.pt
cp ckpt/main.pt ckpt/main_full_12500.pt
THREADS=4 python3 wsp.py ckpt/A.pt ckpt/Astar.pt > logs/wsp_stdout.log 2>&1
cp logs/wsp.log logs/wsp_A.log; cp logs/wsp_report.json logs/wsp_A_report.json
# the "new checkpoint": the same run continued for 1,500 more steps at the floor learning rate
python3 train.py --name main --d 160 --heads 5 --mlp 480 --n_in 2 --n_rec 3 --n_out 2 --max_loops 4 --bs 256 \
  --total_steps 14000 --sched_steps 12500 --warmup 200 --minutes 45 --eval_every 500 --eval_n 3000 > logs/main_seg3.log 2>&1
python3 make_ckpt.py ckpt/main.pt ckpt/B.pt
echo stage1 done
