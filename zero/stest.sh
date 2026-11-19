#!/bin/bash
# One search test, both sides on the same network:
#   stest.sh <label> <new binary> <old binary> [openings=300] [tc=10+0.1] [new options|-] [old options|-] [positions per move|-]
# Each opening is played with both colours; the result (new engine's score, Elo with its 95% range, time losses) is
# appended to zero/search_tests.log. Run it under zero/burst.sh so the self-play workers are paused meanwhile.
cd /root/loopchess; L=zero/search_tests.log; NET=${NET:-zero/st_net.bin}
uo="Hash=64"; [ -n "$6" ] && [ "$6" != "-" ] && uo="$uo,$6"; oo="Hash=64"; [ -n "$7" ] && [ "$7" != "-" ] && oo="$oo,$7"
nd=""; [ -n "$8" ] && [ "$8" != "-" ] && nd="--nodes $8"
r=$(python3 gauntlet.py --us "calc/$2 $NET" --us_opts "$uo" --opp "$3|calc/$3 $NET|$oo" --tc ${5:-10+0.1} --openings ${4:-300} --workers 48 $nd | tail -1 | cut -c1-240)
echo "[$(date -u +%H:%M:%S)] $1: $r" >> $L
