#!/bin/bash
# Where the engine stands, on equal terms: same machine, one thread and 64 MB of hash per engine, every opening played
# with both colours, games decided by the rules or the clock. Three time controls. The network is frozen for the whole test.
cd /root/loopchess; OUT=zero/standing.jsonl; L=zero/standing.log; US="calc/calc_t6 zero/test_net.bin"
F="Fruit 2.1|/usr/games/fruit|Hash=64,OwnBook=false"
G="Glaurung 2.2|/usr/games/glaurung|Hash=64,Threads=1,OwnBook=false"
T="Toga II 3.0|/usr/games/toga2|Hash=64,OwnBook=false"
S14="Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1"
S19="Stockfish 19|/root/sf19/stockfish19|Hash=64,Threads=1"
run() { tc=$1; n=$2; shift 2; for o in "$@"; do python3 gauntlet.py --us "$US" --opp "$o" --tc $tc --openings $n --workers 48 --out $OUT | tail -1 | cut -c1-400 >> $L; done; }
echo "[$(date -u +%H:%M:%S)] start: $(cat zero/test_net.gen 2>/dev/null)" >> $L
run 10+0.1 100 "$F" "$G" "$T" "$S14" "$S19"
run 60+0.6 50 "$F" "$G" "$T" "$S14" "$S19"
run 120+1 25 "$G" "$T" "$S19"
echo "[$(date -u +%H:%M:%S)] finished" >> $L
