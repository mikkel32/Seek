#!/bin/bash
# The standing test again where it was unfair. zero/conc.py: with 48 games side by side, Stockfish 19 keeps about a third
# of the speed it has on a quiet machine while our engine keeps about 60% (its evaluator is ~100 times smaller and needs
# far less memory traffic). So the 48-game test handicapped Stockfish. Here: the same network (generation 27), the same
# openings, one thread and 64 MB each, but only 8 games at a time and ALL self-play paused (checked before every match).
# Glaurung 2.2 is the control: its speed in a crowd falls like ours, so its result should not change.
cd /root/loopchess; OUT=zero/standing_quiet.jsonl; L=zero/standing_quiet.log; US="calc/calc_untuned zero/test_net.bin"
S19="Stockfish 19|/root/sf19/stockfish19|Hash=64,Threads=1"; G="Glaurung 2.2|/usr/games/glaurung|Hash=64,Threads=1,OwnBook=false"
quiet() { for p in $(ps -eo pid,args | awk '$2=="bash" && $3=="zero/worker.sh" {print $1}'); do kill -STOP -$p 2>/dev/null; done; sleep 3
          echo "[$(date -u +%H:%M:%S)] self-play workers still running: $(ps -eo stat,args | awk '$1 !~ /^T/ && $2=="bash" && $3=="zero/worker.sh"' | wc -l); other engine processes: $(ps -eo stat,comm | awk '$1 ~ /^R/ && ($2 ~ /^calc/ || $2 ~ /stockfish/)' | wc -l)" >> $L; }
run() { quiet; python3 gauntlet.py --us "$US" --opp "$3" --tc $1 --openings $2 --workers 8 --out $OUT | tail -1 | cut -c1-400 >> $L; }
echo "[$(date -u +%H:%M:%S)] start (quiet rerun): generation 27 network, 8 games at a time" >> $L
run 10+0.1 100 "$S19"
run 60+0.6 30 "$S19"
for p in $(ps -eo pid,args | awk '$2=="bash" && $3=="zero/worker.sh" {print $1}'); do kill -CONT -$p 2>/dev/null; done
echo "[$(date -u +%H:%M:%S)] finished" >> $L
