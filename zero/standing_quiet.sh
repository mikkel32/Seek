#!/bin/bash
# The standing test again where it was unfair. zero/conc.py: with 48 games side by side, Stockfish 19 keeps about a third
# of the speed it has on a quiet machine while our engine keeps about 60% (its evaluator is ~100 times smaller and needs
# far less memory traffic). So the 48-game test handicapped Stockfish. Here: the same network (generation 27), the same
# openings, one thread and 64 MB each, but only 8 games at a time and all self-play paused.
# Glaurung 2.2 is the control: its speed in a crowd falls like ours, so its result should not change.
cd /root/loopchess; OUT=zero/standing_quiet.jsonl; L=zero/standing_quiet.log; US="calc/calc_untuned zero/test_net.bin"; : > $L
S19="Stockfish 19|/root/sf19/stockfish19|Hash=64,Threads=1"; G="Glaurung 2.2|/usr/games/glaurung|Hash=64,Threads=1,OwnBook=false"
run() { python3 gauntlet.py --us "$US" --opp "$3" --tc $1 --openings $2 --workers 8 --out $OUT | tail -1 | cut -c1-400 >> $L; }
echo "[$(date -u +%H:%M:%S)] start: generation 27 network, 8 games at a time" >> $L
run 10+0.1 100 "$S19"
run 10+0.1 40 "$G"
run 60+0.6 30 "$S19"
echo "[$(date -u +%H:%M:%S)] finished" >> $L
