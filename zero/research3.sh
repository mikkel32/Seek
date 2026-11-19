#!/bin/bash
# The teacher experiment again, on equal terms: the teacher-trained networks rescaled to the production score scale.
cd /root/loopchess; L=zero/research3.log; E=calc/calc_z5
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
K=$(python3 - <<'PY'
import numpy as np, torch, torch.nn.functional as F
x = torch.from_numpy(np.load('data/ev_fsval_x.npy')[:200000].astype(np.int64))
i = np.arange(768); LUT = torch.from_numpy(np.concatenate([((i // 64 + 6) % 12) * 64 + ((i % 64) ^ 56), [768]]).astype(np.int64))
PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64), torch.zeros(1)])
def ev(p):
    sd = torch.load(p, map_location='cpu')
    ao = sd['b'] + F.embedding_bag(x, sd['E'], mode='sum', padding_idx=768); ap = sd['b'] + F.embedding_bag(LUT[x], sd['E'], mode='sum', padding_idx=768)
    lin = sd['c'] + F.embedding_bag(x, sd['S'], mode='sum', padding_idx=768); ph = (PT_W[x].sum(1) / 24).clamp(max=1.0); bk = (((x < 768).sum(1) - 2) // 4).clamp(0, 7)
    return ph * lin[:, 0] + (1 - ph) * lin[:, 1] + sd['cb'][bk] + (ao.clamp(0, 1) ** 2 * sd['wo'][bk]).sum(1) + (ap.clamp(0, 1) ** 2 * sd['wp'][bk]).sum(1)
b = ev('zero/fs_tb.bin.ema.pt')
for n in ('to', 'tn'):
    t = ev(f'zero/fs_{n}.bin.ema.pt'); print(n, round(float((t * b).sum() / (b * b).sum()), 4))     # slope of the teacher network's score against the baseline's
PY
)
log "score scale of the teacher networks relative to the baseline: $(echo $K | tr '\n' ' ')"
KO=$(echo "$K" | awk '$1=="to"{print $2}'); KN=$(echo "$K" | awk '$1=="tn"{print $2}')
python3 zero/rescale_net.py zero/fs_to.bin zero/fs_tos.bin $KO | tee -a $L; python3 zero/rescale_net.py zero/fs_tn.bin zero/fs_tns.bin $KN | tee -a $L
for pair in "tos tb" "tns tb"; do
  set -- $pair
  r=$(python3 match.py --a calc:zero/fs_$1.bin:10000:$E --b calc:zero/fs_$2.bin:10000:$E --openings 1000 --workers 10 2>/dev/null | tail -1)
  log "$1 vs $2 (same score scale): $(echo "$r" | python3 -c "import sys, json; d = json.loads(sys.stdin.read()); print(f\"{d['w']}-{d['d']}-{d['l']}  score {d['score']}  Elo {d['elo']} ({d['lo']} to {d['hi']})\")")"
done
G=$(cut -d" " -f1 zero/current)
r=$(python3 elo_ladder.py --player calc:zero/g$G.bin:100000:$E --sf_time 100000 --start 2300 --openings 20 --levels 3 --workers 10 --out zero/elo_g$G.json 2>/dev/null | tail -1)
log "generation $G rating: $(echo "$r" | python3 -c "import sys, json; d = json.loads(sys.stdin.read()); print(d['rating'], '+-', d['ci95'], '(', d['games'], 'games, node-based)')")"
log finished
