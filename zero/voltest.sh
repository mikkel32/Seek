#!/bin/bash
# One test of the uncertainty-scaled pruning margins (calc_s5 + a network with the reliability read-out), both sides
# searching the same number of positions per move; the other side uses the same margins everywhere.
#   voltest.sh <label> <options for the new side> [positions per move=50000] [openings=1000]
cd /root/loopchess
r=$(python3 gauntlet.py --us "calc/calc_s5 zero/vol_net.bin" --us_opts "Hash=64,$2" --opp "same margins everywhere|calc/calc_s5 zero/vol_net.bin|Hash=64" --tc 10+0.1 --openings ${4:-1000} --workers 48 --nodes ${3:-50000} | tail -1 | cut -c1-240)
echo "[$(date -u +%H:%M:%S)] $1: $r" >> zero/search_tests.log
