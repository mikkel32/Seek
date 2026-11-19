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
# (the transformer at the root of the calculator's search was dropped: one evaluation is worth about 14 calculator positions)
cp zero/ouro.pt zero/ouro_eval.pt 2>/dev/null
while [ "$(ls zero/rawkeep 2>/dev/null | wc -l)" -lt ${NFILES:-420} ]; do sleep 15; done
touch zero/RAW_STOP; sleep 8
G=$(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['gen'])"); START=zero/r17_start.pt; cp zero/g$G.bin.ema.pt $START; PST=$(ls zero/g${G}s.npz zero/g$((G - 1))s.npz 2>/dev/null | head -1)
python3 zero/foresight.py r17 "zero/rawkeep/*.bin" 8 >> $L 2>&1
rm -rf zero/rawkeep
arm() { n=$1; shift
  python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files r17 --ytag yl --val r17val --epochs 3 --lr 1e-3 --threads 4 --device cuda --pst $PST --widen $START "$@" --out zero/r17_$n.bin > zero/r17_$n.train.log 2>&1
  say "arm $n ($*): played held-out $(grep -h '^done' zero/r17_$n.train.log | grep -o 'averaged ([0-9.]*' | cut -c11-18)  $(grep -h 'foresight' zero/r17_$n.train.log | tail -1 | grep -o 'foresight.*' | cut -c1-110)"; }
arm base
arm inv --aux 0.02 --aux_set inv
#  3. Labels from the transformer instead of a search (zero/ouro_label.py): if these train an equally strong evaluator, the
#     graphics card can label several times more positions per hour than the self-play workers do by searching.
python3 zero/ouro_label.py r17 zero/ouro_eval.pt >> $L 2>&1
t0=$PST; python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files r17 --ytag yT --val r17val --epochs 3 --lr 1e-3 --threads 4 --device cuda --pst $PST --widen $START --out zero/r17_tlabel.bin > zero/r17_tlabel.train.log 2>&1
say "arm tlabel (labels from the transformer): played held-out $(grep -h '^done' zero/r17_tlabel.train.log | grep -o 'averaged ([0-9.]*' | cut -c11-18)"
rm -f data/ev_r17_aux.npy data/ev_r17_ys.npy data/ev_r17_y.npy data/ev_r17_s.npy
for t in inv tlabel; do
  echo "run bash zero/gt.sh \"20,000 positions per move, $t vs base\" zero/research17.log --us \"calc/calc_w zero/r17_$t.bin\" --opp \"base|calc/calc_w zero/r17_base.bin|Hash=64\" --openings 1000 --nodes 20000 --stop 0,20" >> zero/search_queue.txt
done
echo "run rm -f data/ev_r17_*.npy; echo \"[\$(date -u +%H:%M:%S)] finished\" >> zero/research17.log" >> zero/search_queue.txt
say "trained; the two matches are in the test queue"
