#!/bin/bash
# The transformer A/B, two arms at a time (each needs one CPU core to drive the card). Starts after the experiments that
# leave little CPU spare; every arm sees the same files (fixed by the first one).
cd /root/loopchess
while ! grep -q "\] finished" zero/search_gap.log 2>/dev/null; do sleep 20; done
S=${STEPS:-20000}
python3 zero/ouro_ab.py base $S > /dev/null 2>&1 &
sleep 90                                                             # the first arm writes the file list
python3 zero/ouro_ab.py smolgen $S smolgen=128 > /dev/null 2>&1 &
wait
python3 zero/ouro_ab.py big $S d=192 heads=6 mlp=576 > /dev/null 2>&1 &
python3 zero/ouro_ab.py aux $S aux=1 > /dev/null 2>&1 &
wait
python3 zero/ouro_ab.py smolgen_aux $S smolgen=128 aux=1 > /dev/null 2>&1
echo "[$(date -u +%H:%M:%S)] all arms finished" >> zero/ouro_ab_base.log
