#!/bin/bash
# timecost.sh <socket> <LmDepth> <LmWait>: one engine, 8 test positions at 300,000 nodes, wall time a search
cd /root/loopchess; S=$1; D=$2; W=$3
python3 -c "
import json
tr = [json.loads(l) for l in open('data/q/truth.jsonl')][100:108]
print('uci'); [print(f'setoption name {k} value {v}') for k, v in (('Hash',64),('PvPrune',1),('ChkExt',0),('SeDouble',20),('CutIir',1),('LmDepth',$D),('PolW',2000),('PolLmr',1),('LmWait',$W))]
print('isready')
for t in tr: print('ucinewgame'); print('position fen ' + t['fen']); print('go nodes 300000')
print('quit')" > /tmp/tc.txt
t0=$(date +%s%N); r=$(OURO_SOCK=$S calc/calc_s19M zero/w_BIGM.bin < /tmp/tc.txt | grep "string transformer" | tail -1 | cut -d" " -f4-)
echo "LmDepth $D LmWait $W: $(( ($(date +%s%N) - t0) / 8000000 )) ms a search; $r"
