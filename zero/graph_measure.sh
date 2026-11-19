#!/bin/bash
# Measurements for the progress chart: the head-to-head matches that restarts lost, and node-based ratings
# (ours and Stockfish's strength levels both at 100,000 nodes per move) for the generations never rated on that ruler.
cd /root/loopchess; E=calc/calc_z5; D=zero/graph_data.txt
h2h() { r=$(python3 match.py --a calc:zero/g$1.bin:4000:$E --b calc:zero/g$2.bin:4000:$E --openings 100 --workers 10 2>/dev/null | tail -1); echo "h2h $1 $2 $r" >> $D; }
rate() { r=$(python3 elo_ladder.py --player calc:$3:100000:$E --sf_time 100000 --start $2 --openings 20 --levels 3 --workers 10 --out zero/elo_n$1.json 2>/dev/null | tail -1); echo "rate $1 $r" >> $D; }
h2h 15 14; h2h 16 15; h2h 18 17; h2h 19 18
( rate 21 2300 zero/g21.bin & rate 1 1400 zero/l1.bin & wait )
( rate 18 2250 zero/g18.bin & rate 2 1500 zero/l2.bin & wait )
( rate 15 2200 zero/g15.bin & rate 3 1800 zero/l3.bin & wait )
rate 4 1900 zero/l4.bin
echo finished >> $D
