#!/bin/bash
# q_screen.sh <engine binary> <positions per move> <log> <option>...     each option (Name=value[,Name=value]) is added to the
# standard options and compared, move by move, with the standard options at the same budget (ext/quality.py).
# QLIM=<n>: only the first n test positions (default 8964, the first collection; later ones are kept for confirmation).
# Locks: zero/Q_BUSY = a test is running (tests queue behind each other); the gap-filling jobs (ext/q_fill.sh, the
# generation farm) stop while it exists.
cd /root/loopchess; E=$1; N=$2; LOG=$3; shift 3; L=${QLIM:-8964}
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; M="$E zero/w_BIGM.bin"
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
[ -e data/q/runs/bigm_$N.json ] || python3 ext/quality.py test --name bigm_$N --cmd "$M" --opts "$B" --nodes $N --limit $L 2>&1 | tail -1 >> $LOG
echo "[$(date -u +%H:%M:%S)] screen at $N positions per move" >> $LOG
for o in "$@"; do
  name=$(echo "$o" | tr -d '=,' )_$N
  python3 ext/quality.py test --name $name --cmd "$M" --opts "$B,$o" --nodes $N --limit $L > /dev/null 2>&1
  r=$(python3 ext/quality.py cmp $name bigm_$N 2>&1 | tail -1 | sed -e 's/^.*difference //' -e 's/; negative = the first gives away less//')
  echo "$o: $r" >> $LOG
done
rm -f zero/Q_BUSY; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE; echo "[$(date -u +%H:%M:%S)] finished" >> $LOG
