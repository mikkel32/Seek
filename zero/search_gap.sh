#!/bin/bash
# Games for zero/search_gap.py: our engine against Stockfish 14.1 playing on its hand-written evaluator (network off),
# at the clock and with both sides searching the same number of positions per move. Starts when no other experiment
# is playing (never two bursts at once).
cd /root/loopchess; L=zero/search_gap.log
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
while ! grep -q "\] finished" zero/${AFTER:-research15}.log 2>/dev/null; do sleep 20; done
G=$(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['gen'])"); cp zero/g$G.bin zero/sg_net.bin
cat > zero/sg_match.sh <<'EOM'
#!/bin/bash
cd /root/loopchess; L=zero/search_gap.log
g() { r=$(python3 gauntlet.py --us "calc/calc_w zero/sg_net.bin" --opp "$1" --tc 10+0.1 --openings 150 --workers 48 $2 --out zero/search_gap.jsonl | tail -1 | cut -c1-260); echo "[$(date -u +%H:%M:%S)] $r" >> $L; }
sleep 10
g "Stockfish 14.1 hand-written evaluator|/usr/games/stockfish|Hash=64,Threads=1,Use NNUE=false"
g "Stockfish 14.1 hand-written evaluator|/usr/games/stockfish|Hash=64,Threads=1,Use NNUE=false" "--nodes 100000"
g "Stockfish 14.1 network|/usr/games/stockfish|Hash=64,Threads=1" "--nodes 100000"
EOM
say "games: generation $G, 300 per pairing"
bash zero/burst.sh 48 bash zero/sg_match.sh
say "finished"
