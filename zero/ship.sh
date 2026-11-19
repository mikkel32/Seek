#!/bin/bash
# Send every new shard of this machine to the machine with the big disk (address and token in /root/.vault). A shard is
# hard-linked the moment it is complete, so the production loop here may delete its own name; it is sent in 48 MB pieces
# to a temporary name there and renamed once its size is right. If more than 10 are waiting, the oldest are dropped.
cd /root/loopchess; . /root/.vault; mkdir -p data/ship
last=$(cat zero/ship.last 2>/dev/null || echo -1)
say() { echo "[$(date -u +%H:%M:%S)] $*" >> ship.log; }
while [ ! -f zero/SHIP_STOP ]; do
  for c in $(ls data/ev_b[0-9]*_c.npy 2>/dev/null); do
    n=$(echo $c | sed 's/.*ev_b\([0-9]*\)_c.npy/\1/'); y=data/ev_b${n}_y.npy
    [ "$n" -le "$last" ] && continue; [ -f "$y" ] || continue; [ -f data/ship/c$n ] && continue
    [ $(( $(date +%s) - $(stat -c %Y $y) )) -lt 20 ] && continue
    ln $c data/ship/c$n 2>/dev/null && ln $y data/ship/y$n 2>/dev/null
  done
  pending=$(ls data/ship/ 2>/dev/null | grep -c '^c')
  if [ "$pending" -gt 40 ]; then for c in $(ls data/ship/c* | sed 's/.*\/c//' | sort -n | head -n $(( pending - 40 ))); do rm -f data/ship/c$c data/ship/y$c; say "dropped shard $c unsent (too many waiting)"; done; fi
  n=$(ls data/ship/ 2>/dev/null | grep '^c' | sed 's/^c//' | sort -n | head -1)
  if [ -z "$n" ]; then sleep 15; continue; fi
  ok=1; t0=$(date +%s)
  for k in c y; do
    f=data/ship/$k$n; dst=/root/loopchess/data/in_l${n}_$k.part; sz=$(stat -c %s $f); off=0; app=""
    while [ $off -lt $sz ]; do
      if dd if=$f bs=1M skip=$(( off / 1048576 )) count=48 2>/dev/null | curl -s -m 240 --fail -H "X-Token: $T3" -X PUT --data-binary @- "$A3/f?p=$dst$app" > /dev/null; then off=$(( off + 48 * 1048576 )); app="&a=1"; else ok=0; break; fi
    done
    [ $ok = 1 ] || break
  done
  if [ $ok = 1 ]; then
    r=$(curl -s -m 60 -H "X-Token: $T3" -X POST --data-binary "cd /root/loopchess/data && [ \$(stat -c %s in_l${n}_c.part) = $(stat -c %s data/ship/c$n) ] && [ \$(stat -c %s in_l${n}_y.part) = $(stat -c %s data/ship/y$n) ] && mv in_l${n}_y.part ev_l${n}_y.npy && mv in_l${n}_c.part ev_l${n}_c.npy && echo shipped" "$A3/run?t=30")
    if echo "$r" | grep -q shipped; then rm -f data/ship/c$n data/ship/y$n; echo $n > zero/ship.last; last=$n; say "shard $n sent in $(( $(date +%s) - t0 ))s; $(( pending - 1 )) waiting"; else say "shard $n: the other side did not confirm ($(echo $r | cut -c1-60))"; sleep 30; fi
  else say "shard $n: sending failed"; sleep 30; fi
done
