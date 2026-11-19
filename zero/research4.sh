#!/bin/bash
# Do the search parameters the engine tuned for itself (20,000 self-play iterations) beat the hand-set ones?
# Same network on both sides, equal nodes, at the self-play budget and at a deeper one.
cd /root/loopchess; L=zero/research4.log; E=calc/calc_z5
T="Razor=293,Rfp=53,NmpEv=191,LmpBase=6,FutBase=89,FutMul=94,HistPrune=4245,SeeQ=48,SeeC=92,SeMul=2,Asp=33,DeltaQ=190,LmrBase=68,LmrDiv=240,HbQuad=5,HbLin=33,HbMax=1322"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
G=$(cut -d" " -f1 zero/current); cp zero/g$G.bin zero/tune_net.bin
for n in 2500 10000; do
  r=$(python3 match.py --a calc:zero/tune_net.bin:$n:$E:$T --b calc:zero/tune_net.bin:$n:$E --openings 1000 --workers 10 2>/dev/null | tail -1)
  log "self-tuned vs hand-set search, generation $G network, $n nodes: $(echo "$r" | python3 -c "import sys, json; d = json.loads(sys.stdin.read()); print(f\"{d['w']}-{d['d']}-{d['l']}  score {d['score']}  Elo {d['elo']} ({d['lo']} to {d['hi']})\")")"
done
log finished
