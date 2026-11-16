#!/bin/bash
# The one-pass mixed-data probe (Lichess + engine-game positions, consistent labels) at the clock on the match machine:
# against BIGK directly, and its standing against Stockfish 14.1 at both time controls. A preview of the full-scale run.
cd /root/loopchess; N=${1:-p_mixc}; . /root/.vault
[ -s zero/$N.bin ] || curl -s --fail -m 300 -H "X-Token: $T3" "$A3/f?p=/root/loopchess/zero/$N.bin" -o zero/$N.bin
while pgrep -f "[p]robe_clock.sh" > /dev/null; do sleep 10; done; sleep 5
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
g() { bash gt.sh "$1" mixc_clock.log 76 --us "calc/calc_s19 zero/$N.bin" --us_opts "$B" "${@:2}"; }
g "$N (1.09 billion mixed positions, one pass) against BIGK (4.37 billion Lichess, four passes), 10+0.1" --opp "BIGK|calc/calc_s19 zero/w_BIGK.bin|$B" --tc 10+0.1 --openings 250
g "standing, $N: against Stockfish 14.1, 10+0.1" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --tc 10+0.1 --openings 250
g "standing, $N: against Stockfish 14.1, 60+0.6" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --tc 60+0.6 --openings 150
g "$N against BIGK, 60+0.6" --opp "BIGK|calc/calc_s19 zero/w_BIGK.bin|$B" --tc 60+0.6 --openings 100
echo "[$(date -u +%H:%M:%S)] finished" >> mixc_clock.log
