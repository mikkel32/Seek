#!/bin/bash
# Hash-size control at 300,000 nodes: does the 19% move instability come from collisions?  512 MB and 1 GB tables
# (practically no collisions at this budget) against each other and against 64 MB; 8 engines at once (memory)
cd /root/loopchess; B="PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; LOG=hash_ctl.log
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY
for h in 512 1024; do
  python3 ext/quality.py test --name hctl_$h --cmd "calc/calc_s19M zero/w_BIGM.bin" --opts "Hash=$h,$B" --nodes 300000 --limit 8964 --workers 8 > /dev/null 2>&1
done
python3 - >> $LOG <<'PY'
import json
r = {n: json.load(open(f'data/q/runs/{n}.json')) for n in ('hctl_512', 'hctl_1024', 'bigm_300000')}
m = {n: dict(zip(x['fens'], x['moves'])) for n, x in r.items()}
def flips(a, b):
    k = [f for f in m[a] if f in m[b]]; return 100 * sum(m[a][f] != m[b][f] for f in k) / len(k), len(k)
for a, b in (('hctl_512', 'hctl_1024'), ('hctl_512', 'bigm_300000'), ('hctl_1024', 'bigm_300000')):
    p, n = flips(a, b); print(f'{a} against {b}: another move in {p:.1f}% of {n} positions')
PY
for x in hctl_512 hctl_1024; do echo "$x against 64 MB: $(python3 ext/quality.py cmp $x bigm_300000 2>&1 | tail -1 | sed -e 's/^.*difference //' -e 's/; negative = the first gives away less//')" >> $LOG; done
rm -f zero/Q_BUSY; echo "[$(date -u +%H:%M:%S)] finished" >> $LOG
