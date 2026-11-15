#!/bin/bash
# restart the continuous loop on its current settings (the running cycle finishes first; self-play resumes within seconds)
cd /root/loopchess
N0=$(grep -c "continuous loop stopped" zero/loop.log)
touch zero/HALT
until [ "$(grep -c "continuous loop stopped" zero/loop.log)" -gt "$N0" ]; do sleep 2; done
rm -f zero/HALT
tmux kill-window -t cont:loop 2>/dev/null
tmux new-window -d -t cont -n loop "cd /root/loopchess && NPROC=56 NPLAY=48 NODES=2500 FILE_GAMES=200 MIN_NEW=48000 ${LOOP_ENV:-} python3 zero/cont.py 2>&1 | tee -a zero/cont_stdout.log"
