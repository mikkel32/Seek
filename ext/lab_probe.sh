#!/bin/bash
# Architecture probes on the mixed data, on the match machine's card (the training machine's is busy with the full run):
# the reference recipe (768 wide, 16 king buckets, one pass over Lichess shards h0-h33 + engine-game shards s0-s33 with
# the rescaled result-mixed labels) with one change each. Each network then plays the reference probe from the training
# machine (p_mixc) at 30,000 positions per move and Stockfish 14.1 at 10,000 positions per move each, 2,000 games both.
#   bash lab_probe.sh "<name>|<trainer options>" ...
cd /root/loopchess; . /root/.vault; mkdir -p /dev/shm/d
[ -s zero/p_mixc.bin ] || curl -s --fail -m 300 -H "X-Token: $T3" "$A3/f?p=/root/loopchess/zero/p_mixc.bin" -o zero/p_mixc.bin
one() { f=$1; [ -s /dev/shm/d/$f ] && return; off=0; : > /dev/shm/d/$f.part; ok=1
    while true; do
      got=0; for try in 1 2 3 4; do curl -s --fail -m 300 -H "X-Token: $T3" "$A3/f?p=/root/loopchess/data/$f&o=$off&n=48000000" -o /dev/shm/d/$f.piece && { got=1; break; }; sleep 3; done
      [ $got = 1 ] || { ok=0; break; }
      sz=$(stat -c %s /dev/shm/d/$f.piece); cat /dev/shm/d/$f.piece >> /dev/shm/d/$f.part; off=$(( off + sz )); [ $sz -lt 48000000 ] && break
    done
    rm -f /dev/shm/d/$f.piece; [ $ok = 1 ] && mv /dev/shm/d/$f.part /dev/shm/d/$f && ln -sf /dev/shm/d/$f data/$f; }
export -f one; export A3 T3
for k in $(seq 34 59); do rm -f /dev/shm/d/ev_h${k}_c.npy /dev/shm/d/ev_h${k}_y.npy; done          # not part of the mixed probe: make room in memory
for k in $(seq 0 33); do ln -sf ev_h${k}_y.npy data/ev_h${k}_yc.npy; echo ev_s${k}_c.npy; echo ev_s${k}_yc.npy; done | xargs -P 6 -I{} bash -c "one {}"
echo "[$(date -u +%H:%M:%S)] fetched: $(ls /dev/shm/d/ev_s*_c.npy | wc -l) engine-game shards, $(du -sh /dev/shm/d | cut -f1) in memory" >> lab_probe.log
tags=$(seq -s, -f "h%g" 0 33),$(seq -s, -f "s%g" 0 33); B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
for spec in "$@"; do
  n=${spec%%|*}; o=${spec#*|}; t0=$(date +%s)
  python3 calc/train_nnue.py --h 768 --buckets 8 --kb 16 --kf 1 --ema 0.999 --files $tags --val ival --ytag yc --epochs 1 --bs 32768 --lr 1.4e-3 --threads 6 --device cuda --pst zero/g101s.npz --out zero/p_$n.bin $o > zero/p_$n.train.log 2>&1
  r=$(grep "^done" zero/p_$n.train.log | sed 's/.*averaged (\([0-9.]*\), \([0-9.]*\)).*/held-out loss \1, mean error \2/')
  echo "[$(date -u +%H:%M:%S)] $n ($o): ${r:-FAILED: $(tail -1 zero/p_$n.train.log | cut -c1-120)}; $(( $(date +%s) - t0 ))s, $(grep '^step' zero/p_$n.train.log | tail -1 | grep -o '[0-9]*k pos/s')" >> lab_probe.log
  [ -z "$r" ] && continue
  echo "[$(date -u +%H:%M:%S)]   engine against trainer: $(python3 zero/check_net.py calc/calc_s19 zero/p_$n.bin zero/p_$n.bin.ema.pt 1000 2>&1 | tail -1 | cut -c1-200)" >> lab_probe.log
  for m in p_mixc p_$n; do echo "[$(date -u +%H:%M:%S)]   speed, $m: $(python3 zero/bench.py calc/calc_s19 zero/$m.bin 13 | cut -c1-90)" >> lab_probe.log; done
  bash gt.sh "  $n against the reference probe, 30,000 positions per move" lab_probe.log 76 --us "calc/calc_s19 zero/p_$n.bin" --us_opts "$B" --opp "p_mixc|calc/calc_s19 zero/p_mixc.bin|$B" --openings 1000 --nodes 30000
  bash gt.sh "  $n against Stockfish 14.1, 10,000 positions per move each" lab_probe.log 76 --us "calc/calc_s19 zero/p_$n.bin" --us_opts "$B" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --openings 1000 --nodes 10000
done
echo "[$(date -u +%H:%M:%S)] probes finished" >> lab_probe.log
