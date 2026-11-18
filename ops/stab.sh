#!/bin/bash
# how often does a meaningless change (another hash size) change the move?  ours and Stockfish, 300,000 nodes
cd /root/loopchess
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
B="PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
q() { python3 ext/quality.py "$@" 2>&1 | tail -1 >> stab.log; }
q test --name bigm_h32_300000 --cmd "calc/calc_s19h zero/w_BIGM.bin" --opts "Hash=32,$B" --nodes 300000 --limit 8964
q cmp bigm_h32_300000 bigm_300000
q test --name sf_h32_300000 --cmd /usr/games/stockfish --opts "Hash=32,Threads=1" --nodes 300000 --limit 8964
q cmp sf_h32_300000 sf_300000
python3 - >> stab.log <<'PY'
import json
D = 'data/q/runs/'
for a, b in (('bigm_h32_300000', 'bigm_300000'), ('sf_h32_300000', 'sf_300000'), ('bigm_100000', 'bigm_300000'), ('sf_100000', 'sf_300000'), ('bigm_300000', 'bigm_1000000'), ('sf_300000', 'sf_1000000')):
    ra, rb = json.load(open(D + a + '.json')), json.load(open(D + b + '.json')); mb = dict(zip(rb['fens'], rb['moves']))
    n = sum(1 for f in ra['fens'] if f in mb); d = sum(1 for f, m in zip(ra['fens'], ra['moves']) if f in mb and mb[f] != m)
    print(f'{a} and {b}: a different move in {d} of {n} positions = {100 * d / n:.1f}%')
PY
rm -f zero/Q_BUSY; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE; echo "[$(date -u +%H:%M:%S)] finished" >> stab.log
