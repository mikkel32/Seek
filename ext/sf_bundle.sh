#!/bin/bash
# sf_bundle.sh <positions per move> <log> "<name>|<options>"...     groups of search settings on the search-only yardstick:
# our search on Stockfish 14.1's own network (-DSFNET build, EvalScale=140) against the same build with the standing
# options (run sfnet_<N> of ext/quality.py); Stockfish itself at the same budget is run sf_<N>.
cd /root/loopchess; N=$1; LOG=$2; shift 2; export SFNET_FILE=/root/ext/sf141_embedded.bin
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1,EvalScale=140"; M="calc/calc_s19sf2 zero/w_BIGM.bin"
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
echo "[$(date -u +%H:%M:%S)] our search on Stockfish's network, $N positions per move; difference to the standing options" >> $LOG
for g in "$@"; do name=${g%%|*}; o=${g#*|}
  python3 ext/quality.py test --name sfb_${name}_$N --cmd "$M" --opts "$B,$o" --nodes $N --limit 8964 > /dev/null 2>&1
  r=$(python3 ext/quality.py cmp sfb_${name}_$N sfnet_$N 2>&1 | tail -1 | sed -e 's/^.*difference //' -e 's/; negative = the first gives away less//')
  r2=$(python3 ext/quality.py cmp sfb_${name}_$N sf_$N 2>&1 | tail -1 | sed -e 's/^.*difference //' -e 's/ (standard.*$//')
  echo "$name [$o]: $r; against Stockfish itself $r2" >> $LOG
done
rm -f zero/Q_BUSY; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE; echo "[$(date -u +%H:%M:%S)] finished" >> $LOG
