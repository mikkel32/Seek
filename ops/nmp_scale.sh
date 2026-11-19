#!/bin/bash
# the null-move gate from the hash table, and the per-unit integer scale, on the quality test at 300,000 nodes (BIGM)
cd /root/loopchess
while ! grep -q finished scale_match.log 2>/dev/null; do sleep 10; done
bash ext/q_screen.sh calc/calc_s19h 300000 screen300k.log NmpTt=1 "NmpTt=1,NmpMargin=44" ThrRfp=60
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
python3 ext/quality.py test --name bigm_pu_300000 --cmd "calc/calc_s19q zero/w_BIGM.bin" --opts "$B" --nodes 300000 --limit 8964 > /dev/null 2>&1
echo "per-unit integer scale (BIGM): $(python3 ext/quality.py cmp bigm_pu_300000 bigm_300000 2>&1 | tail -1 | sed -e 's/^.*difference //' -e 's/; negative = the first gives away less//')" >> screen300k.log
rm -f zero/Q_BUSY; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE; echo "[$(date -u +%H:%M:%S)] finished nmp_scale" >> screen300k.log
