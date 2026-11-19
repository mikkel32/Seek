#!/bin/bash
# Two-brain teacher on the long-memory loop: does a label that blends in OURO's judgement make a stronger fast evaluator?
# (On the short-memory loop, with a younger OURO, it was worth +20 Elo against the same recipe without OURO.)
# The same games, three labels in the same score scale (zero/teacher2.py): production's (y95), the teacher recipe without
# OURO (yN, the control) and with OURO (yO). Three evaluators from the same start, the same steps; 2,000 games per pairing
# at 20,000 positions per move, plus 300 clock games for the one that matters.
cd /root/loopchess; L=zero/research12.log; : > $L
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
while [ "$(ls zero/rawkeep | wc -l)" -lt ${NFILES:-700} ]; do sleep 15; done
touch zero/RAW_STOP; sleep 8
G=$(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['gen'])")
START=zero/r12_start.pt; cp zero/g$G.bin.ema.pt $START; PST=$(ls zero/g${G}s.npz zero/g$((G - 1))s.npz 2>/dev/null | head -1)
cp data/ev_shval_x.npy data/ev_r12val_x.npy; cp data/ev_shval_y.npy data/ev_r12val_y.npy
say "start: generation $G weights, $(ls zero/rawkeep | wc -l) raw files, OURO at $(tail -1 zero/ouro.log | grep -o 'step [0-9]*')"
python3 zero/teacher2.py r12 zero/rawkeep $START zero/ouro.pt 8 2>&1 | grep -v "positions judged" >> $L
rm -rf zero/rawkeep
for t in y95 yN yO; do
  t0=$(date +%s); python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files r12 --ytag $t --val r12val --epochs ${EP:-3} --lr 1e-3 --threads 4 --device cuda --pst $PST --widen $START --out zero/r12_$t.bin > zero/r12_$t.train.log 2>&1
  say "arm $t: $(grep -h '^positions' zero/r12_$t.train.log | cut -c1-40) train loss $(grep -h '^step' zero/r12_$t.train.log | tail -1 | grep -o 'loss [0-9.]*')  [$(( $(date +%s) - t0 ))s]"
done
for t in y95 yN yO; do say "arm $t evaluates the opening position at $(printf 'position startpos\neval\nquit\n' | calc/calc_w zero/r12_$t.bin | grep '^eval')"; done
cat > zero/r12_match.sh <<'EOM'
#!/bin/bash
cd /root/loopchess; L=zero/research12.log
m() { r=$(python3 match.py --a calc:zero/r12_$1.bin:20000:calc/calc_w --b calc:zero/r12_$2.bin:20000:calc/calc_w --openings 1000 --workers 48 | tail -1 | cut -c1-110); echo "[$(date -u +%H:%M:%S)] 20,000 positions per move, $1 vs $2: $r" >> $L; }
m yO y95; m yN y95; m yO yN
r=$(python3 gauntlet.py --us "calc/calc_w zero/r12_yO.bin" --opp "y95|calc/calc_w zero/r12_y95.bin|Hash=64" --tc 10+0.1 --openings 150 --workers 48 --out zero/research12.jsonl | tail -1 | cut -c1-150)
echo "[$(date -u +%H:%M:%S)] clock 10+0.1, yO vs y95: $r" >> $L
EOM
bash zero/burst.sh 48 bash zero/r12_match.sh
say "finished"
