#!/bin/bash
# one engine, 8 test positions, the transformer steering without waiting: requests and answers per search
cd /root/loopchess
python3 -c "
import json
tr = [json.loads(l) for l in open('data/q/truth.jsonl')][:8]
print('uci'); [print(f'setoption name {k} value {v}') for k, v in (('Hash',64),('PvPrune',1),('ChkExt',0),('SeDouble',20),('CutIir',1),('LmDepth',8),('PolW',2000),('PolLmr',1),('LmWait',${1:-0}))]
print('isready')
for t in tr: print('ucinewgame'); print('position fen ' + t['fen']); print('go nodes 300000')
print('quit')" > /tmp/rc.txt
t0=$(date +%s%N); OURO_SOCK=$2 calc/calc_s19M zero/w_BIGM.bin < /tmp/rc.txt | grep "string" | awk '{print $5, $8, $12}' | tr "\n" ";"; echo " (requests, used, late, cumulative) $(( ($(date +%s%N) - t0) / 1000000 )) ms for 8 searches"
