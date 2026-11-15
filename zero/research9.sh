#!/bin/bash
# (1) which training label is closest to strong-play truth (zero/label_truth.py, 8,000 positions)
# (2) does the loop with the long memory hold its gain at the clock? The newest generation (at least the third trained on
#     the 32-shard window) against generation 35, where the 6-shard loop had been stuck, and against X3 (research8's
#     one-off retraining on 18 shards).
cd /root/loopchess; L=zero/research9.log; : > $L
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
while ! grep -q "^finished" zero/label_truth_smoke.log 2>/dev/null; do sleep 10; done
if grep -q "^truth = " zero/label_truth_smoke.log; then
  bash zero/burst.sh 44 python3 zero/label_truth.py 8000 44 > zero/label_truth.log 2> zero/label_truth.err; say "label comparison done"
else say "label comparison skipped: the smoke test failed: $(tail -2 zero/label_truth_smoke.log | head -1 | cut -c1-200)"; fi
until [ "$(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['gen'])")" -ge 62 ]; do sleep 20; done
G=$(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['gen'])"); cp zero/g$G.bin zero/r9_now.bin
say "long-memory loop: generation $G"
cat > zero/r9_match.sh <<'EOM'
#!/bin/bash
cd /root/loopchess; L=zero/research9.log
g() { r=$(python3 gauntlet.py --us "calc/calc_untuned $1" --opp "$3|calc/calc_untuned $2|Hash=64" --tc $4 --openings $5 --workers 48 --out zero/research9.jsonl | tail -1 | cut -c1-150); echo "[$(date -u +%H:%M:%S)] clock $4, $6: $r" >> $L; }
g zero/r9_now.bin zero/r5_S.bin "generation 35" 10+0.1 200 "long-memory generation vs generation 35"
g zero/r9_now.bin zero/keep_X3.bin "X3" 10+0.1 150 "long-memory generation vs X3"
g zero/r9_now.bin zero/r5_S.bin "generation 35" 60+0.6 60 "long-memory generation vs generation 35"
EOM
bash zero/burst.sh 48 bash zero/r9_match.sh
say "finished"
