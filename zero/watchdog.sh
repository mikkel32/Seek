#!/bin/bash
# keeps production alive: if the continuous loop is not running (and was not halted on purpose) for 90 seconds, start it.
# Stop the watchdog with: touch zero/WATCHDOG_STOP
cd /root/loopchess
miss=0
while [ ! -f zero/WATCHDOG_STOP ]; do
  if [ ! -f zero/HALT ] && [ -z "$(ps -eo args | awk '$1=="python3" && $2=="zero/cont.py"')" ]; then miss=$((miss+1)); else miss=0; fi
  if [ $miss -ge 3 ]; then
    echo "[$(date -u +%H:%M:%S)] watchdog: the loop was not running; starting it" >> zero/loop.log
    tmux new-window -d -t cont -n loopw "cd /root/loopchess && NPROC=56 NPLAY=48 NODES=2500 FILE_GAMES=200 MIN_NEW=48000 python3 zero/cont.py 2>&1 | tee -a zero/cont_stdout.log"
    miss=0; sleep 120
  fi
  sleep 30
done
