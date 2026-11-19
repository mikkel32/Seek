#!/bin/bash
# Should the evaluator learn from a different label?
# zero/label_truth.py: against strong-play truth, production's label (lambda-return, lambda 0.95) is further away than the
# evaluator's own output; lambda 0.7 and the plain search score are much closer (0.335 and 0.337 against 0.374).
# Here the SAME games are labelled four ways; four evaluators start from the same weights, train the same number of steps,
# and play each other. Same shape and speed, so games at a fixed 20,000 positions per move decide (2,000 per pairing), with
# 300 clock games as a check.
cd /root/loopchess; L=zero/research11.log; : > $L
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
while ! grep -q "finished" zero/research10.log 2>/dev/null; do sleep 15; done
while [ "$(ls zero/rawkeep | wc -l)" -lt ${NFILES:-1200} ] && [ ! -e zero/RAW_STOP ]; do sleep 15; done
touch zero/RAW_STOP; sleep 8
python3 zero/make_targets.py r11 zero/rawkeep 8 >> $L 2>&1
G=$(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['gen'])")
START=zero/r11_start.pt; cp zero/g$G.bin.ema.pt $START; PST=$(ls zero/g${G}s.npz zero/g$((G - 1))s.npz 2>/dev/null | head -1)
cp data/ev_shval_x.npy data/ev_r11val_x.npy; cp data/ev_shval_y.npy data/ev_r11val_y.npy
say "start: generation $G weights"
for t in y95 y70 y50 y00; do
  t0=$(date +%s); python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files r11 --ytag $t --val r11val --epochs ${EP:-2} --lr 1e-3 --threads 4 --device cuda --pst $PST --widen $START --out zero/r11_$t.bin > zero/r11_$t.train.log 2>&1
  say "arm $t: $(grep -h '^positions' zero/r11_$t.train.log | cut -c1-40) train loss $(grep -h '^step' zero/r11_$t.train.log | tail -1 | grep -o 'loss [0-9.]*')  [$(( $(date +%s) - t0 ))s]"
done
cat > zero/r11_match.sh <<'EOM'
#!/bin/bash
cd /root/loopchess; L=zero/research11.log
for t in y70 y50 y00; do
  r=$(python3 match.py --a calc:zero/r11_$t.bin:20000:calc/calc_w --b calc:zero/r11_y95.bin:20000:calc/calc_w --openings 1000 --workers 48 | tail -1 | cut -c1-110)
  echo "[$(date -u +%H:%M:%S)] 20,000 positions per move, $t vs y95: $r" >> $L
done
for t in y70 y50 y00; do
  r=$(python3 gauntlet.py --us "calc/calc_w zero/r11_$t.bin" --opp "y95|calc/calc_w zero/r11_y95.bin|Hash=64" --tc 10+0.1 --openings 150 --workers 48 --out zero/research11.jsonl | tail -1 | cut -c1-150)
  echo "[$(date -u +%H:%M:%S)] clock 10+0.1, $t vs y95: $r" >> $L
done
EOM
bash zero/burst.sh 48 bash zero/r11_match.sh
say "finished"
