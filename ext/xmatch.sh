#!/bin/bash
# As each network trained on outside data is ready, play it against the self-taught reference (N384) at equal positions per move.
cd /root/loopchess
for pair in "LE:lichess.log:Stockfish-labelled Lichess positions" "SF1:sfdata.log:Stockfish's own training data"; do
  n=${pair%%:*}; rest=${pair#*:}; log=${rest%%:*}; what=${rest#*:}
  while ! grep -q "finished" $log 2>/dev/null; do sleep 30; done
  echo "[$(date -u +%H:%M:%S)] speed, $n: $(python3 zero/bench.py calc/calc_s17 zero/w_$n.bin 13 | cut -c1-90)" >> ext.log
  touch zero/FARM_PAUSE; sleep 5
  bash gt.sh "network trained on $what ($n) against the self-taught network (N384), 30,000 positions per move" ext.log 22 --us "calc/calc_s17 zero/w_$n.bin" --opp "N384 self-taught|calc/calc_s17 zero/w_N384.bin|Hash=64" --openings 1000 --nodes 30000
  rm -f zero/FARM_PAUSE
done
echo "[$(date -u +%H:%M:%S)] finished" >> ext.log
