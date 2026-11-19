#!/bin/bash
# keep every shard the loop makes (hard links: no copy; the loop's own files come and go as before) so experiments can
# train on far more games than the loop's sliding window. Stops when the disk gets short.
cd /root/loopchess
while [ ! -e zero/ARCHIVE_STOP ]; do
  free=$(df --output=avail -BG . | tail -1 | tr -dc 0-9)
  [ "$free" -lt 12 ] && { echo "[$(date -u +%H:%M:%S)] disk down to ${free}G: archiving stopped" >> zero/archive.log; exit 0; }
  last=$(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['shards'][-1])" 2>/dev/null)
  for f in data/ev_sh*_x.npy; do
    k=${f#data/ev_sh}; k=${k%_x.npy}
    [ -n "$last" ] && [ "$k" -le "$last" ] && [ ! -e data/ev_ar${k}_x.npy ] && [ -e data/ev_sh${k}_y.npy ] && ln data/ev_sh${k}_y.npy data/ev_ar${k}_y.npy && ln $f data/ev_ar${k}_x.npy && echo "[$(date -u +%H:%M:%S)] kept shard $k" >> zero/archive.log
  done
  sleep 30
done
