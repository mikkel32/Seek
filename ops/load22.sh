#!/bin/bash
# 22 engines at once, 3 searches each, transformer steering without waiting
cd /root/loopchess; S=$1
for i in $(seq 1 22); do
  printf "uci\nsetoption name Hash value 64\nsetoption name LmDepth value 8\nsetoption name PolW value 2000\nsetoption name PolLmr value 1\nsetoption name LmWait value 0\nisready\nposition fen 2r2rk1/1b2qppp/p3pn2/1p6/3N4/P1N1P3/1P2QPPP/2RR2K1 w - - 0 18\ngo nodes 300000\nposition fen r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R2QKB1R w KQ - 0 8\ngo nodes 300000\nposition startpos moves e2e4 c7c5 g1f3 d7d6\ngo nodes 300000\nquit\n" | OURO_SOCK=$S timeout 60 calc/calc_s19M zero/w_BIGM.bin 2>&1 | grep string | tail -1 > /tmp/load_$i.txt &
done
wait; cat /tmp/load_*.txt | sort | uniq -c | sort -rn | head -5
