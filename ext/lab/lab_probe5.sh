#!/bin/bash
# Second round of probes on the match machine's card, each with its own shard list: first-stage Stockfish data in the mix,
# 32 king buckets on the mixed data, 1024 wide on the mixed data. Same recipe and matches as lab_probe.sh.
#   bash lab_probe2.sh "<name>|<trainer options>|<shards, e.g. h:0-22,s:0-22,t:0-22>" ...
cd /root/loopchess; . /root/.vault
one() { f=$1; g=${2:-$1}; [ -s /dev/shm/d/$g ] && return; off=0; : > /dev/shm/d/$g.part; ok=1
    while true; do
      got=0; for try in 1 2 3 4; do curl -s --fail -m 300 -H "X-Token: $T3" "$A3/f?p=/root/loopchess/data/$f&o=$off&n=48000000" -o /dev/shm/d/$g.piece && { got=1; break; }; sleep 3; done
      [ $got = 1 ] || { ok=0; break; }
      sz=$(stat -c %s /dev/shm/d/$g.piece); cat /dev/shm/d/$g.piece >> /dev/shm/d/$g.part; off=$(( off + sz )); [ $sz -lt 48000000 ] && break
    done
    rm -f /dev/shm/d/$g.piece; [ $ok = 1 ] && mv /dev/shm/d/$g.part /dev/shm/d/$g && ln -sf /dev/shm/d/$g data/$g; }
export -f one; export A3 T3
for k in $(seq 0 22); do echo "ev_t${k}_c.npy ev_t${k}_c.npy"; echo "ev_t${k}_ym.npy ev_t${k}_yc.npy"; done | xargs -P 6 -L 1 bash -c 'one $0 $1'
echo "[$(date -u +%H:%M:%S)] fetched: $(ls /dev/shm/d/ev_t*_c.npy | wc -l) first-stage shards, $(du -sh /dev/shm/d | cut -f1) in memory" >> lab_probe.log
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
for spec in "$@"; do
  IFS='|' read n o g <<< "$spec"; t0=$(date +%s)
  tg=""; for r in ${g//,/ }; do pre=${r%%:*}; rg=${r#*:}; tg="$tg,$(seq -s, -f "$pre%g" ${rg%-*} ${rg#*-})"; done; tg=${tg#,}
  python3 calc/train_nnue.py --h 768 --buckets 8 --kb 16 --kf 1 --ema 0.999 --files $tg --val ival --ytag yc --epochs 1 --bs 32768 --lr 1.4e-3 --threads 6 --device cuda --pst zero/g101s.npz --out zero/p_$n.bin $o > zero/p_$n.train.log 2>&1
  r=$(grep "^done" zero/p_$n.train.log | sed 's/.*averaged (\([0-9.]*\), \([0-9.]*\)).*/held-out loss \1, mean error \2/')
  echo "[$(date -u +%H:%M:%S)] $n ($o; $g): ${r:-FAILED: $(tail -1 zero/p_$n.train.log | cut -c1-120)}; $(( $(date +%s) - t0 ))s, $(grep '^step' zero/p_$n.train.log | tail -1 | grep -o '[0-9]*k pos/s')" >> lab_probe.log
  [ -z "$r" ] && continue
  echo "[$(date -u +%H:%M:%S)]   engine against trainer: $(python3 zero/check_net.py calc/calc_s19x zero/p_$n.bin zero/p_$n.bin.ema.pt 1000 2>&1 | tail -1 | cut -c1-200)" >> lab_probe.log
  bash gt.sh "  $n against the reference probe, 30,000 positions per move" lab_probe.log 76 --us "calc/calc_s19x zero/p_$n.bin" --us_opts "$B" --opp "p_mixc|calc/calc_s19x zero/p_mixc.bin|$B" --openings 1000 --nodes 30000
  bash gt.sh "  $n against Stockfish 14.1, 10,000 positions per move each" lab_probe.log 76 --us "calc/calc_s19x zero/p_$n.bin" --us_opts "$B" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --openings 1000 --nodes 10000
done
echo "[$(date -u +%H:%M:%S)] second round of probes finished" >> lab_probe.log
# the wide probe at the clock against the reference probe (after the node-count sweep has left the processors)
while pgrep -f "[d]eep_sweep.sh" > /dev/null; do sleep 20; done
bash gt.sh "  mixc_w1536 against the reference probe, 10+0.1" lab_probe.log 76 --us "calc/calc_s19x zero/p_mixc_w1536.bin" --us_opts "$B" --opp "p_mixc|calc/calc_s19x zero/p_mixc.bin|$B" --tc 10+0.1 --openings 500
bash gt.sh "  mixc_w1536 against the reference probe, 60+0.6" lab_probe.log 76 --us "calc/calc_s19x zero/p_mixc_w1536.bin" --us_opts "$B" --opp "p_mixc|calc/calc_s19x zero/p_mixc.bin|$B" --tc 60+0.6 --openings 150
