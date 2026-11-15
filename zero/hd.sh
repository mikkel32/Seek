#!/bin/bash
# how much does the layered read-out lose at equal positions per move, precisely, and does it depend on the depth of search?
cd /root/loopchess
g() { bash gt.sh "$1" head.log 76 --us "calc/calc_s13 zero/w_N384H.bin" --opp "N384|calc/calc_s13 zero/w_N384.bin|Hash=64" "${@:2}"; }
g "layered read-out, precise: N384H against N384, 30,000 positions per move, 2,000 games" --openings 1000 --nodes 30000
g "layered read-out: N384H against N384, 5,000 positions per move, 2,000 games" --openings 1000 --nodes 5000
g "layered read-out: N384H against N384, 150,000 positions per move, 1,000 games" --openings 500 --nodes 150000
echo "[$(date -u +%H:%M:%S)] finished" >> head.log
