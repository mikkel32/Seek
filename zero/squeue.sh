#!/bin/bash
# Runs the lines of zero/search_queue.txt one at a time (arguments of zero/stest.sh), each under a burst that pauses the
# self-play workers. Lines can be appended while it runs. Waits first for the experiment named in $AFTER to finish, so
# two bursts never overlap. Stops when zero/SQ_STOP exists.
cd /root/loopchess
while ! grep -q "\] finished" zero/${AFTER:-search_gap}.log 2>/dev/null; do sleep 20; done
G=$(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['gen'])"); [ -e zero/st_net.bin ] || cp zero/g$G.bin zero/st_net.bin
n=0
while [ ! -e zero/SQ_STOP ]; do
  if [ $n -lt $(wc -l < zero/search_queue.txt) ]; then
    n=$((n + 1)); line=$(sed -n "${n}p" zero/search_queue.txt)
    [ -n "$line" ] && bash zero/burst.sh 48 bash zero/stest.sh $line
  else sleep 15; fi
done
