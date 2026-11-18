#!/bin/bash
# the hash table: two entries per slot against one, and how stable the choice is under a 1% change of the node budget
cd /root/loopchess
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
B="PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; M="calc/calc_s19h zero/w_BIGM.bin"
q() { python3 ext/quality.py "$@" 2>&1 | tail -1 >> stab.log; }
q test --name bigm_tb_300000 --cmd "$M" --opts "Hash=64,$B,TtBucket=1" --nodes 300000 --limit 8964
q cmp bigm_tb_300000 bigm_300000
q test --name bigm_tb_h32_300000 --cmd "$M" --opts "Hash=32,$B,TtBucket=1" --nodes 300000 --limit 8964
q cmp bigm_tb_h32_300000 bigm_tb_300000
q test --name bigm_h512_300000 --cmd "$M" --opts "Hash=512,$B" --nodes 300000 --limit 8964
q cmp bigm_h512_300000 bigm_300000
q test --name bigm_303000 --cmd "$M" --opts "Hash=64,$B" --nodes 303000 --limit 8964
q test --name sf_303000 --cmd /usr/games/stockfish --opts "Hash=64,Threads=1" --nodes 303000 --limit 8964
python3 - >> stab.log <<'PY'
import json
D = 'data/q/runs/'
for a, b in (('bigm_tb_300000', 'bigm_300000'), ('bigm_tb_h32_300000', 'bigm_tb_300000'), ('bigm_h512_300000', 'bigm_300000'), ('bigm_303000', 'bigm_300000'), ('sf_303000', 'sf_300000')):
    ra, rb = json.load(open(D + a + '.json')), json.load(open(D + b + '.json')); mb = dict(zip(rb['fens'], rb['moves']))
    n = sum(1 for f in ra['fens'] if f in mb); d = sum(1 for f, m in zip(ra['fens'], ra['moves']) if f in mb and mb[f] != m)
    print(f'{a} and {b}: a different move in {d} of {n} positions = {100 * d / n:.1f}%')
PY
rm -f zero/Q_BUSY; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE; echo "[$(date -u +%H:%M:%S)] finished 2" >> stab.log
