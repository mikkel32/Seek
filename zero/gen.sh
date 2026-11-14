#!/bin/bash
# One chunk of one generation of self-play: NPROC processes, each writing its own record file.
#   zero/gen.sh <generation> <weights|mob> <games per process> <nodes per move> <adjudicate 0|1> [tree positions per move] [chunk]
cd "$(dirname "$0")/.."
G=$1; W=$2; N=$3; NODES=$4; ADJ=$5; TREE=${6:-0}; C=${7:-0}
ENGINE=${ENGINE:-calc/calc_z3}; NPROC=${NPROC:-4}
for i in $(seq 0 $((NPROC - 1))); do
  rm -f zero/g${G}_c${C}_p$i.bin
  $ENGINE $W selfplay zero/g${G}_c${C}_p$i.bin $N $NODES $((G * 100000 + C * 1000 + i + 1)) $ADJ $TREE > zero/g${G}_c${C}_p$i.log 2>&1 &
done
wait
tail -q -n 1 zero/g${G}_c${C}_p*.log | tail -4
