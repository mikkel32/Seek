#!/bin/bash
# Copy shards of the outside data from the machine with the big disk into this machine MEMORY (/dev/shm/d, linked into
# data/), so that this machine graphics card can train probe networks. The other machine serves at most 50 MB a request:
# a file is fetched in 48 MB ranges. Address and token of the other machine: /root/.vault.
#   bash fetch_shards.sh <tag> ...
cd /root/loopchess; . /root/.vault; mkdir -p /dev/shm/d
one() { t=$1; for k in c y; do f=ev_${t}_$k.npy; [ -s /dev/shm/d/$f ] && continue
    off=0; : > /dev/shm/d/$f.part; ok=1
    while true; do
      got=0; for try in 1 2 3 4; do curl -s --fail -m 300 -H "X-Token: $T3" "$A3/f?p=/root/loopchess/data/$f&o=$off&n=48000000" -o /dev/shm/d/$f.piece && { got=1; break; }; sleep 3; done
      [ $got = 1 ] || { ok=0; break; }
      sz=$(stat -c %s /dev/shm/d/$f.piece); cat /dev/shm/d/$f.piece >> /dev/shm/d/$f.part; off=$(( off + sz )); [ $sz -lt 48000000 ] && break
    done
    rm -f /dev/shm/d/$f.piece; [ $ok = 1 ] && mv /dev/shm/d/$f.part /dev/shm/d/$f && ln -sf /dev/shm/d/$f data/$f; done; }
export -f one; export A3 T3
printf "%s\n" "$@" | xargs -P 6 -I{} bash -c "one {}"
echo "[$(date -u +%H:%M:%S)] fetched: $(ls /dev/shm/d/*_c.npy | wc -l) shards, $(du -sh /dev/shm/d | cut -f1)" >> fetch.log

