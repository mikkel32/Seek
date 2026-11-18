#!/bin/bash
# one-off: at the old trainer's next checkpoint (step 20000), stop it and start the new one under the restart wrapper
cd /root/loopchess
until grep -q "step 20000 " zero/ouro.log; do sleep 5; done
sleep 3; touch zero/OURO_STOP
until tail -1 zero/ouro.log | grep -q "OURO trainer stopped"; do sleep 2; done
tmux kill-window -t cont:ouro 2>/dev/null
rm -f zero/OURO_STOP
tmux new-window -d -t cont -n ouro "bash zero/ouro_loop.sh"
