#!/bin/bash
# total positions per second of N copies of the bench at once (our engine), and of Stockfish
cd /root/loopchess; touch zero/GEN_PAUSE; sleep 3
one() { python3 zero/bench.py $1 $2 15 | grep -o "= [0-9]*k" | grep -o "[0-9]*"; }; export -f one
a=$(one calc/calc_s19c zero/w_BIGM.bin)
l=$(seq 22 | xargs -P 22 -I{} bash -c "one calc/calc_s19c zero/w_BIGM.bin" | awk '{s+=$1} END {print s}')
l2=$(seq 12 | xargs -P 12 -I{} bash -c "one calc/calc_s19c zero/w_BIGM.bin" | awk '{s+=$1} END {print s}')
sfone() { (echo "setoption name Hash value 64"; echo "position startpos"; echo "go nodes 4000000"; sleep 7; echo quit) | /usr/games/stockfish | grep -o "nps [0-9]*" | tail -1 | grep -o "[0-9]*"; }; export -f sfone
s=$(seq 22 | xargs -P 22 -I{} bash -c sfone | awk '{s+=$1} END {print int(s/1000)}')
rm -f zero/GEN_PAUSE
echo "BIGM alone ${a}k; 22 copies in total ${l}k; 12 copies in total ${l2}k; Stockfish 14.1, 22 copies in total ${s}k"
