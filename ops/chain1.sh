#!/bin/bash
cd /root/loopchess
while ! grep -q finished calib.log; do sleep 10; done
bash ext/q_screen.sh calc/calc_s19e 100000 screen100k.log \
 PvPrune=0 SeDouble=0 CutIir=0 ChkExt=1 SearchV3=1 \
 CutLmr=1 CutLmr=2 ProbCut=150 NegExt=1 TtCapLmr=1 LmrPrune=1 LmpDepth=8 LmpDepth=63 NmpDiv=3 NmpBase=4 NmpTt=1 \
 PruneV2=1 PruneV2=2 PruneV2=4 PruneV2=8 PruneV2=16 HistV2=1 HistV2=2 HistV2=4 HistV2=8 HistV2=16 \
 LmrDiv=215 LmrDiv=270 LmrBase=48 LmrBase=88 LmrHist=5000 LmrHist=12000 \
 Rfp=43 Rfp=65 FutMul=75 FutMul=115 SeeQ=35 SeeQ=65 HistPrune=3000 HistPrune=6000 Razor=230 Razor=360 NmpEv=150 NmpEv=240 \
 SeDepth=6 SeDepth=8 SeMul=1 SeMul=3 SeDouble=10 SeDouble=35 \
 Deeper=0 CapLmr=0 Asp=20 Asp=50 DeltaQ=150 DeltaQ=240 LmpBase=4 LmpBase=8
