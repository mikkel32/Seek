#!/bin/bash
# The 768-wide network trained on all 4.37 billion Lichess positions (BIGW) at the clock: against the 768-wide one trained on
# 0.79 billion (LFW), then a standing against Stockfish 14.1 with and without its network at both time controls.
# The search self-tuning is stopped for the duration and resumed afterwards with BIGW as its network.
cd /root/loopchess
touch zero/SPSA_STOP; while pgrep -f "[s]psa_clock" > /dev/null; do sleep 2; done; rm -f zero/SPSA_STOP; sleep 8
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
for n in LFW BIGW; do echo "[$(date -u +%H:%M:%S)] speed, $n: $(python3 zero/bench.py calc/calc_s17 zero/w_$n.bin 14 | cut -c1-90)" >> bw.log; done
g() { bash gt.sh "$1" bw.log 76 --us "calc/calc_s17 zero/w_BIGW.bin" "${@:2}"; }
g "BIGW (4.37 billion positions) against LFW (0.79 billion), both 768 wide, at the clock, 10+0.1" --opp "LFW|calc/calc_s17 zero/w_LFW.bin|Hash=64" --tc 10+0.1 --openings 200
g "standing, BIGW with the confirmed options: against Stockfish 14.1, 10+0.1" --us_opts "$B" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --tc 10+0.1 --openings 200
g "standing, BIGW with the confirmed options: against Stockfish 14.1 on its hand-written evaluator, 10+0.1" --us_opts "$B" --opp "Stockfish 14.1 hand-written evaluator|/usr/games/stockfish|Hash=64,Threads=1,Use NNUE=false" --tc 10+0.1 --openings 200
g "standing, BIGW with the confirmed options: against Stockfish 14.1, 60+0.6" --us_opts "$B" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --tc 60+0.6 --openings 150
g "BIGW (4.37 billion positions) against LFW (0.79 billion), both 768 wide, at the clock, 60+0.6" --opp "LFW|calc/calc_s17 zero/w_LFW.bin|Hash=64" --tc 60+0.6 --openings 100
g "standing, BIGW with the confirmed options: against Stockfish 14.1 on its hand-written evaluator, 60+0.6" --us_opts "$B" --opp "Stockfish 14.1 hand-written evaluator|/usr/games/stockfish|Hash=64,Threads=1,Use NNUE=false" --tc 60+0.6 --openings 100
echo "[$(date -u +%H:%M:%S)] finished" >> bw.log
tmux new-window -d -t lab -n spsa "ENGINE=calc/calc_s17 SPSA_PROCS=76 SPSA_STATE=zero/spsa_lfw.json SPSA_FIXED=PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1 SPSA_EXTRA=MmCorr:128:0:512:40,ThrRfp:500:0:2000:150 python3 zero/spsa_clock.py zero/w_BIGW.bin 15000 20+0.2 >> spsa_lfw.log 2>&1"

