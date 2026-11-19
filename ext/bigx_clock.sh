#!/bin/bash
# The 1536-wide full-scale network (BIGX) at the clock on the match machine, against BIGM and against Stockfish 14.1.
# Waits for the training machine to finish it, fetches it, then plays with the usual 76 boards at once and with 44
# (one game per physical core: under 76 boards the wide network loses more speed than the narrow one).
# Every match takes the processor lock, so probe matches on this machine never overlap a clock match.
#   bash bigx_clock.sh [network stem=w_BIGX]
cd /root/loopchess; N=${1:-w_BIGX}; . /root/.vault
get() { f=$1; off=0; : > $f.part
    while true; do
      got=0; for try in 1 2 3 4; do curl -s --fail -m 300 -H "X-Token: $T3" "$A3/f?p=/root/loopchess/$f&o=$off&n=48000000" -o $f.piece && { got=1; break; }; sleep 3; done
      [ $got = 1 ] || return 1
      sz=$(stat -c %s $f.piece); cat $f.piece >> $f.part; off=$(( off + sz )); [ $sz -lt 48000000 ] && break
    done
    rm -f $f.piece; mv $f.part $f; }
while true; do
  curl -s --fail -m 60 -H "X-Token: $T3" "$A3/f?p=/root/loopchess/big_run.log" -o /dev/shm/big_run.vault 2>/dev/null
  grep -q "network ${N#w_} trained" /dev/shm/big_run.vault 2>/dev/null && break; sleep 30
done
sleep 20; get zero/$N.bin || { echo "[$(date -u +%H:%M:%S)] could not fetch $N" >> bigx_clock.log; exit 1; }
get zero/$N.bin.ema.pt
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; SF="Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1"; BM="BIGM|calc/calc_s19 zero/w_BIGM.bin|$B"
g() { w=$1; l=$2; shift 2; flock /dev/shm/cpu.lock bash gt.sh "$l" bigx_clock.log $w --us "calc/calc_s19x zero/$N.bin" --us_opts "$B" "$@"; }
echo "[$(date -u +%H:%M:%S)] fetched $N ($(stat -c %s zero/$N.bin) bytes); speed alone: $(flock /dev/shm/cpu.lock python3 zero/bench.py calc/calc_s19x zero/$N.bin 14 | cut -c1-90); BIGM: $(flock /dev/shm/cpu.lock python3 zero/bench.py calc/calc_s19 zero/w_BIGM.bin 14 | cut -c1-90)" >> bigx_clock.log
g 76 "$N against BIGM, 10+0.1, 76 boards at once" --opp "$BM" --tc 10+0.1 --openings 250
g 76 "standing, $N: against Stockfish 14.1, 10+0.1, 76 boards at once" --opp "$SF" --tc 10+0.1 --openings 250
g 44 "$N against BIGM, 10+0.1, 44 boards at once" --opp "$BM" --tc 10+0.1 --openings 250
g 76 "$N against BIGM, 60+0.6, 76 boards at once" --opp "$BM" --tc 60+0.6 --openings 100
g 76 "standing, $N: against Stockfish 14.1, 60+0.6, 76 boards at once" --opp "$SF" --tc 60+0.6 --openings 150
echo "[$(date -u +%H:%M:%S)] finished" >> bigx_clock.log
