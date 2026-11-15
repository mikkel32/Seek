#!/bin/bash
# Give the loop a long memory. research8: the same evaluator trained on 18 kept shards beats the production weights by 37 Elo
# at the clock, while the loop with its 6-shard window has not moved in 28 generations. The kept shards go back into the
# window, which now holds 32 shards (the new defaults in zero/cont.py). Nothing else changes.
cd /root/loopchess
touch zero/ARCHIVE_STOP zero/WATCHDOG_STOP
N0=$(grep -c "continuous loop stopped" zero/loop.log); touch zero/HALT
until [ "$(grep -c "continuous loop stopped" zero/loop.log)" -gt "$N0" ]; do sleep 2; done
tmux kill-window -t cont:loop 2>/dev/null; tmux kill-window -t cont:loopw 2>/dev/null; tmux kill-window -t cont:watchdog 2>/dev/null; tmux kill-window -t cont:arch 2>/dev/null
for f in data/ev_r5s*_x.npy data/ev_ar*_x.npy; do
  k=$(echo $f | sed -E 's/.*ev_(r5s|ar)([0-9]+)_x.npy/\2/'); y=${f%_x.npy}_y.npy
  [ -e data/ev_sh${k}_x.npy ] || ln $f data/ev_sh${k}_x.npy
  [ -e data/ev_sh${k}_y.npy ] || ln $y data/ev_sh${k}_y.npy
done
python3 - <<PY
import json, glob, re, os
st = json.load(open('zero/cont_state.json'))
have = sorted(int(m.group(1)) for m in (re.search(r"ev_sh(\d+)_x\.npy$", f) for f in glob.glob("data/ev_sh*_x.npy")) if m)
st['shards'] = [k for k in have if k < st['next_shard'] and os.path.exists(f'data/ev_sh{k}_y.npy')]
json.dump(st, open('zero/cont_state.json', 'w')); print('state:', st)
PY
rm -f data/ev_r5s*_[xy].npy data/ev_ar*_[xy].npy          # only the extra names go; the data stays under the loop's names
sed -i 's/MIN_NEW=48000 \(\${LOOP_ENV\)/MIN_NEW=48000 WINDOW=32 WINDOW_POS=480000000 SAMPLE_CAP=240000000 \1/; s/MIN_NEW=48000 python3/MIN_NEW=48000 WINDOW=32 WINDOW_POS=480000000 SAMPLE_CAP=240000000 python3/' zero/restart_loop.sh zero/watchdog.sh
rm -f zero/HALT zero/WATCHDOG_STOP zero/ARCHIVE_STOP
echo "[$(date -u +%H:%M:%S)] the window now holds up to 32 shards (was 6); kept shards restored" >> zero/loop.log
tmux new-window -d -t cont -n loop "cd /root/loopchess && NPROC=60 NPLAY=54 NODES=2500 FILE_GAMES=200 MIN_NEW=48000 WINDOW=32 WINDOW_POS=480000000 SAMPLE_CAP=240000000 python3 zero/cont.py 2>&1 | tee -a zero/cont_stdout.log"
tmux new-window -d -t cont -n watchdog "bash zero/watchdog.sh"
