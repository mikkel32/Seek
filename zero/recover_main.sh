#!/bin/bash
# Bring production back after the machine was restarted: the loop (self-play on the new engine + training), its watchdog,
# the shard archiver and the janitor for the transformer's game records. Safe to run when some of them are already up.
cd /root/loopchess
tmux has-session -t cont 2>/dev/null || tmux new-session -d -s cont -n shell
rm -f zero/HALT zero/WATCHDOG_STOP zero/ARCHIVE_STOP zero/JANITOR_STOP
if [ -z "$(ps -eo args | awk '$1=="python3" && $2=="zero/cont.py"')" ]; then
  tmux new-window -d -t cont -n loop "cd /root/loopchess && NPROC=60 NPLAY=54 NODES=2500 FILE_GAMES=200 MIN_NEW=48000 WINDOW=32 WINDOW_POS=480000000 SAMPLE_CAP=240000000 ENGINE=calc/calc_z6 python3 zero/cont.py 2>&1 | tee -a zero/cont_stdout.log"
fi
pgrep -f "watchdo[g].sh" > /dev/null || tmux new-window -d -t cont -n watchdog "bash zero/watchdog.sh"
pgrep -f "compact.py archiv[e]" > /dev/null || tmux new-window -d -t cont -n arc "nice -n 19 python3 zero/compact.py archive 40 >> zero/archive.log 2>&1"
pgrep -f "janito[r].sh" > /dev/null || tmux new-window -d -t cont -n janitor "bash zero/janitor.sh"
sleep 20
echo "loop: $(ps -eo args | awk '$1=="python3" && $2=="zero/cont.py"' | wc -l)  self-play workers: $(ps -eo args | awk '$1=="bash" && $2=="zero/worker.sh"' | wc -l)  generation $(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['gen'])")"
tail -2 zero/loop.log | cut -c1-160
