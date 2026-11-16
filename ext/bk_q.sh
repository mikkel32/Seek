#!/bin/bash
# The king-bucketed network (BIGK, engine calc_s18) at the clock: against the plain 768-wide network (BIGW) on the same
# positions, then the standing against Stockfish 14.1 at both time controls. The search self-tuning is stopped for the
# duration and resumed afterwards with the network named as the first argument (default BIGK) and calc_s18.
cd /root/loopchess; NEXT=${1:-BIGK}; . /root/.vault
[ -s zero/w_BIGK.bin ] || curl -s --fail -m 200 -H "X-Token: $T3" "$A3/f?p=/root/loopchess/zero/w_BIGK.bin" -o zero/w_BIGK.bin
touch zero/SPSA_STOP; while pgrep -f "[s]psa_clock" > /dev/null; do sleep 2; done; rm -f zero/SPSA_STOP; sleep 8
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
for n in BIGW BIGK; do echo "[$(date -u +%H:%M:%S)] speed, $n: $(python3 zero/bench.py calc/calc_s18 zero/w_$n.bin 14 | cut -c1-90)" >> bk.log; done
g() { bash gt.sh "$1" bk.log 76 --us "calc/calc_s18 zero/w_BIGK.bin" "${@:2}"; }
g "BIGK (16 king buckets) against BIGW (plain), both 768 wide on 4.37 billion positions, at the clock, 10+0.1" --opp "BIGW|calc/calc_s18 zero/w_BIGW.bin|Hash=64" --tc 10+0.1 --openings 200
g "standing, BIGK with the confirmed options: against Stockfish 14.1, 10+0.1" --us_opts "$B" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --tc 10+0.1 --openings 200
g "standing, BIGK with the confirmed options: against Stockfish 14.1, 60+0.6" --us_opts "$B" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --tc 60+0.6 --openings 150
g "BIGK (16 king buckets) against BIGW (plain), both 768 wide on 4.37 billion positions, at the clock, 60+0.6" --opp "BIGW|calc/calc_s18 zero/w_BIGW.bin|Hash=64" --tc 60+0.6 --openings 100
echo "[$(date -u +%H:%M:%S)] finished" >> bk.log
tmux new-window -d -t lab -n spsa "ENGINE=calc/calc_s18 SPSA_PROCS=76 SPSA_STATE=zero/spsa_lfw.json SPSA_FIXED=PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1 SPSA_EXTRA=MmCorr:128:0:512:40,ThrRfp:500:0:2000:150 python3 zero/spsa_clock.py zero/w_$NEXT.bin 15000 20+0.2 >> spsa_lfw.log 2>&1"
