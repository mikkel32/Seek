#!/bin/bash
# Two brains at play time, equal nodes: the calculator (generation 9) with OURO correcting its root-move scores,
# against the same calculator alone. Both search 10,000 nodes per move; OURO's time is not counted here
# (one batch of about 35 positions per move), so this measures the VALUE of its judgment, not the cost.
cd /root/loopchess; L=zero/duov_exp.log; E=calc/calc_z5
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
while ! grep -q "production restarted" zero/nodes_exp.log; do sleep 10; done
[ -f zero/ouro_snap.pt ] || cp zero/ouro.pt zero/ouro_snap.pt
log "OURO snapshot: step $(python3 -c "import torch; print(torch.load('zero/ouro_snap.pt', map_location='cpu').get('step'))" 2>/dev/null | tail -1)"
for v in "1.0:100" "0.5:60" "2.0:150"; do
  grep -q "^$v " zero/duov_res.txt 2>/dev/null && continue
  r=$(python3 match.py --a duov:zero/g9.bin:zero/ouro_snap.pt:${TEST:-10000}:$v:$E --b calc:zero/g9.bin:${TEST:-10000}:$E --openings 300 --workers 20 2>/dev/null | tail -1)
  echo "$v $r" >> zero/duov_res.txt
  log "scale:cap $v  two brains vs calculator alone: $(echo "$r" | python3 -c "import sys, json; d = json.loads(sys.stdin.read()); print(f\"{d['w']}-{d['d']}-{d['l']}  score {d['score']}  Elo {d['elo']} ({d['lo']} to {d['hi']})\")")"
done
log finished
