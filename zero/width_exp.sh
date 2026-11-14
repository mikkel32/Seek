#!/bin/bash
# Width test at equal data: a 512-wide network against production's 256-wide generation 9,
# both grown from generation 8 and trained on exactly the same shards for the same number of epochs.
cd /root/loopchess
while ! grep -q "generation 9 published" zero/loop.log; do sleep 10; done
python3 calc/train_nnue.py --h 512 --buckets 8 --ema 0.999 --files sh0,sh1,sh2,sh3,sh4 --val shval --epochs 2.0 --lr 1e-3 --threads 2 --device cuda \
  --pst zero/g9s.npz --widen zero/g8.bin.ema.pt --out zero/w9_512.bin > zero/width_train.log 2>&1
grep "^done" zero/width_train.log | cut -c1-160 > zero/width_exp.log
while ! grep -q "production restarted" zero/nodes_exp.log; do sleep 10; done      # matches only after the timed arms have finished
for n in 4000 20000; do
  python3 match.py --a calc:zero/w9_512.bin:$n:calc/calc_z5 --b calc:zero/g9.bin:$n:calc/calc_z5 --openings 300 --workers 6 2>/dev/null | tail -1 | sed "s/^/512 vs 256 at $n nodes: /" >> zero/width_exp.log
done
echo finished >> zero/width_exp.log
