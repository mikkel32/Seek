#!/bin/bash
# The transformer A/B: plain against Smolgen, the same positions, the same steps, both at once on the graphics card.
# Waits until self-play has written enough fresh games (the files in zero/opool are read, not removed).
cd /root/loopchess
while [ "$(ls zero/opool | wc -l)" -lt ${MINFILES:-420} ]; do sleep 30; done
S=${STEPS:-20000}
python3 zero/ouro_ab.py base $S > /dev/null 2>&1 &
sleep 150
python3 zero/ouro_ab.py smolgen $S smolgen=128 > /dev/null 2>&1 &
wait
echo "[$(date -u +%H:%M:%S)] both arms finished" >> zero/ouro_ab_base.log
