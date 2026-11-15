#!/bin/bash
# Two ideas from the "bake search into the weights / draft and verify" list, retested properly.
#  1. Draft and verify: the calculator searches as usual, the transformer (on the graphics card) judges the root position
#     once per move and its move preferences are added to the root scores. It did nothing with the young transformer;
#     today's has trained twenty times longer. Both sides search 20,000 positions per move, 200 games per setting.
#  2. Multi-ply prediction for the fast evaluator: extra outputs that must predict what happens later in the game (the
#     search score 2/4/8 plies on, plies left, material changes, the move played), used in training only. At weights 0.15
#     and 0.5 this lost 181 and 277 Elo: at those weights the extra losses were as large as the main one. Retested at
#     0.02 and 0.005 (a few per cent of the main loss). Needs game-ordered raw files; matches go through the test queue.
cd /root/loopchess; L=zero/research17.log; : > $L
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
while ! grep -q "\] finished" zero/research16.log 2>/dev/null; do sleep 20; done          # the card is busy until then
G=$(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['gen'])"); cp zero/g$G.bin zero/r17_net.bin; cp zero/ouro.pt zero/ouro_eval.pt
for lam in 15 40; do
  r=$(OURO_GPU=1 python3 match.py --a duo:zero/r17_net.bin:zero/ouro_eval.pt:20000:$lam:40:p:calc/calc_w --b calc:zero/r17_net.bin:20000:calc/calc_w --openings 100 --workers 6 2>/dev/null | tail -1 | cut -c1-120)
  say "calculator + transformer at the root ($lam centipawns per unit of preference) vs calculator alone, 20,000 positions per move: $r"
done
while [ "$(ls zero/rawkeep 2>/dev/null | wc -l)" -lt ${NFILES:-700} ]; do sleep 15; done
touch zero/RAW_STOP; sleep 8
G=$(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['gen'])"); START=zero/r17_start.pt; cp zero/g$G.bin.ema.pt $START; PST=$(ls zero/g${G}s.npz zero/g$((G - 1))s.npz 2>/dev/null | head -1)
python3 zero/foresight.py r17 "zero/rawkeep/*.bin" 8 >> $L 2>&1
rm -rf zero/rawkeep
arm() { n=$1; shift
  python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files r17 --ytag yl --val r17val --epochs 3 --lr 1e-3 --threads 4 --device cuda --pst $PST --widen $START "$@" --out zero/r17_$n.bin > zero/r17_$n.train.log 2>&1
  say "arm $n ($*): played held-out $(grep -h '^done' zero/r17_$n.train.log | grep -o 'averaged ([0-9.]*' | cut -c11-18)  $(grep -h 'foresight' zero/r17_$n.train.log | tail -1 | grep -o 'foresight.*' | cut -c1-110)"; }
arm base
arm a02 --aux 0.02
arm a005 --aux 0.005
rm -f data/ev_r17_aux.npy data/ev_r17_ys.npy data/ev_r17_y.npy data/ev_r17_s.npy
for t in a02 a005; do
  echo "run r=\$(python3 match.py --a calc:zero/r17_$t.bin:20000:calc/calc_w --b calc:zero/r17_base.bin:20000:calc/calc_w --openings 1000 --workers 48 | tail -1 | cut -c1-110); echo \"[\$(date -u +%H:%M:%S)] 20,000 positions per move, $t vs base: \$r\" >> zero/research17.log" >> zero/search_queue.txt
done
echo "run rm -f data/ev_r17_*.npy; echo \"[\$(date -u +%H:%M:%S)] finished\" >> zero/research17.log" >> zero/search_queue.txt
say "trained; the two matches are in the test queue"
