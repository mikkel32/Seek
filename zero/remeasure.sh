#!/bin/bash
# Generations 3, 4 and 15 were being rated while the engine binary was swapped to the self-tuned search, so their numbers
# mix two engines. Rate them again with the hand-set search (the engine every other point on the chart used), and rate the
# test network with both, to show what the search tuning is worth on this ruler.
cd /root/loopchess; D=zero/graph_data2.txt
rate() { r=$(python3 elo_ladder.py --player calc:$3:100000:$4 --sf_time 100000 --start $2 --openings 20 --levels 3 --workers 10 --out zero/elo_r$1.json 2>/dev/null | tail -1); echo "rate $1 $r" >> $D; }
( rate 3 1800 zero/l3.bin calc/calc_untuned & rate 4 1900 zero/l4.bin calc/calc_untuned & rate 15 2250 zero/g15.bin calc/calc_untuned & wait )
( rate 27u 2300 zero/test_net.bin calc/calc_untuned & rate 27t 2350 zero/test_net.bin calc/calc_t6 & wait )
echo finished >> $D
