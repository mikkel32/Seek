#!/bin/bash
# gt.sh <label> <log file> <gauntlet.py arguments...>     one match on 48 boards at once, the result appended to the log
cd /root/loopchess; label=$1; log=$2; shift 2
r=$(python3 gauntlet.py "$@" --workers 48 | tail -1 | cut -c1-460)
echo "[$(date -u +%H:%M:%S)] $label: $r" >> $log
