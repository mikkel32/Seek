#!/bin/bash
# the five evaluators of research5, judged again on held-out positions that carry the SAME kind of target they were trained on
# (positions from inside the search trees included), taken from a newer shard none of them has seen
cd /root/loopchess; L=zero/research6a.log; : > $L
K=$(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['shards'][-1])")
ln -f data/ev_sh${K}_x.npy data/ev_r6n_x.npy; ln -f data/ev_sh${K}_y.npy data/ev_r6n_y.npy
python3 - <<PY
import numpy as np
x = np.load('data/ev_r6n_x.npy', mmap_mode='r'); y = np.load('data/ev_r6n_y.npy', mmap_mode='r')
i = np.sort(np.random.RandomState(5).choice(len(x), 1_000_000, replace=False))
np.save('data/ev_r6val_x.npy', x[i]); np.save('data/ev_r6val_y.npy', y[i].astype(np.float32) / 255)
print('validation: 1,000,000 positions of shard $K (of', len(x), ')')
PY
echo "same-target held-out set from shard $K" >> $L
PST=$(ls zero/g35s.npz | head -1)
for spec in "S 384 1 zero/r5_start.pt" "A 384 1 zero/r5_A.bin.ema.pt" "B 768 1 zero/r5_B.bin.ema.pt" "C 1536 1 zero/r5_C.bin.ema.pt" "D 384 11 zero/r5_D.bin.ema.pt" "E 384 7 zero/r5_E.bin.ema.pt"; do
  set -- $spec
  for v in r6val r5val; do
    r=$(python3 calc/train_nnue.py --h $2 --kb $3 --buckets 8 --files r5s26 --val $v --epochs 0 --resume $4 --device cuda --pst $PST --out /tmp/r6_scratch.bin 2>&1 | grep -h "^positions" | grep -o "start val.*" | cut -c1-80)
    echo "arm $1 (hidden $2, king regions $3) on $v: $r" >> $L
  done
done
echo finished >> $L
