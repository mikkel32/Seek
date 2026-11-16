#!/bin/bash
# The full-scale mixed-data network (BIGM) at the clock on the match machine: against BIGK directly and its standing against
# Stockfish 14.1 at both time controls. Everything else on the machine's processors is stopped first.
#   bash bigm_clock.sh [network stem=w_BIGM] [extra options for it, e.g. EvalScale=90]
cd /root/loopchess; N=${1:-w_BIGM}; X=${2:-}; . /root/.vault
curl -s --fail -m 300 -H "X-Token: $T3" "$A3/f?p=/root/loopchess/zero/$N.bin" -o zero/$N.bin.part && mv zero/$N.bin.part zero/$N.bin
for w in sw sw2; do tmux kill-window -t lab:$w 2>/dev/null; done; sleep 1; for p in $(pgrep -f "[g]auntlet.py.*lab_sweep|[g]t.sh.*lab_sweep"); do kill $p 2>/dev/null; done
for p in $(pgrep -f "[g]auntlet.py.*positions per move"); do kill $p 2>/dev/null; done; sleep 5
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
g() { bash gt.sh "$1" bigm_clock.log 76 --us "calc/calc_s19 zero/$N.bin" --us_opts "$B${X:+,$X}" "${@:2}"; }
echo "[$(date -u +%H:%M:%S)] speed, $N: $(python3 zero/bench.py calc/calc_s19 zero/$N.bin 14 | cut -c1-90)" >> bigm_clock.log
g "$N${X:+ [$X]} against BIGK, 10+0.1" --opp "BIGK|calc/calc_s19 zero/w_BIGK.bin|$B" --tc 10+0.1 --openings 250
g "standing, $N${X:+ [$X]}: against Stockfish 14.1, 10+0.1" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --tc 10+0.1 --openings 250
g "standing, $N${X:+ [$X]}: against Stockfish 14.1, 60+0.6" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --tc 60+0.6 --openings 150
g "$N${X:+ [$X]} against BIGK, 60+0.6" --opp "BIGK|calc/calc_s19 zero/w_BIGK.bin|$B" --tc 60+0.6 --openings 100
echo "[$(date -u +%H:%M:%S)] finished" >> bigm_clock.log
