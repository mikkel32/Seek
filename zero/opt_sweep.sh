#!/bin/bash
# Optimiser comparison for the fast evaluator's input matrix: same data, same initial weights, same schedule;
# every optimiser gets a learning-rate sweep. Metric: validation log-loss with one-expert routing (lower is better).
cd "$(dirname "$0")/.."
OUT=logs/opt_sweep.txt
run() { name=$1; shift
  grep -q "^$name " $OUT 2>/dev/null && return
  r=$(python3 calc/train_nnue.py --h 128 --buckets 8 --files z3t --max_rows 6000000 --val z3v --epochs 2 --pst zero/g4s.npz --threads 3 --seed 1 --out zero/sweep_$name.bin "$@" 2>&1 | grep "^done" | sed 's/.*raw (\([0-9.]*\), \([0-9.]*\)).*/\1 \2/')
  echo "$name $r" >> $OUT; }
run adam_1e-3   --opt adam --lr 1e-3
run adam_2e-3   --opt adam --lr 2e-3
run adam_5e-4   --opt adam --lr 5e-4
for k in muon2 normuon muon; do for m in 0.01 0.03 0.1; do run ${k}_$m --opt $k --mlr $m --lr 1e-3; done; done
cat $OUT
