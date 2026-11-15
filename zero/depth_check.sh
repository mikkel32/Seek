#!/bin/bash
# The ladder (100,000 nodes per move, against Stockfish levels) shows generation 21 no higher than generation 12, while the
# generation-to-generation matches (4,000 nodes) keep showing gains. Play the two generations against each other at both
# budgets: if the newer one wins clearly at 4,000 nodes but not at 100,000, the gains do not carry to deeper search.
cd /root/loopchess; E=calc/calc_z5; L=zero/depth_check.log
for n in 4000 100000; do
  r=$(python3 match.py --a calc:zero/g21.bin:$n:$E --b calc:zero/g12.bin:$n:$E --openings 150 --workers 10 2>/dev/null | tail -1)
  echo "[$(date -u +%H:%M:%S)] generation 21 vs 12 at $n nodes: $(echo "$r" | python3 -c "import sys, json; d = json.loads(sys.stdin.read()); print(f\"{d['w']}-{d['d']}-{d['l']}  score {d['score']}  Elo {d['elo']} ({d['lo']} to {d['hi']})  depth {d['a_blocks_per_move']:.1f} vs {d['b_blocks_per_move']:.1f}\")")" | tee -a $L
done
echo finished >> $L
