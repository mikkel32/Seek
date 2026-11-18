#!/bin/bash
cd /root/loopchess
while pgrep -f "rec_matc[h].sh" > /dev/null; do sleep 10; done
touch zero/Q_BUSY
python3 ext/autopsy.py score games_bigs.jsonl 100000 22 > autopsy.log 2>&1
python3 ext/autopsy.py report games_bigs.jsonl >> autopsy.log 2>&1
rm -f zero/Q_BUSY; echo "[$(date -u +%H:%M:%S)] finished" >> autopsy.log
