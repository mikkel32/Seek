#!/bin/bash
# The production loop on one machine: self-play never stops (zero/farm2.py); every STEP new shards the next generation is
# trained from the previous one on the newest KEEP shards and published to the self-play workers.
#   bash zero/prod2.sh <starting network .bin (with its .pt next to it)>        stop: touch zero/PROD_STOP
#   matches that need the processors: touch zero/FARM_PAUSE ... rm zero/FARM_PAUSE
cd /root/loopchess; W=${W:-80}; KEEP=${KEEP:-36}; STEP=${STEP:-4}; EP=${EP:-0.5}
say() { echo "[$(date -u +%H:%M:%S)] $*" >> prod2.log; }
top() { ls data/ev_b[0-9]*_c.npy | sed 's/.*ev_b\([0-9]*\)_c.npy/\1/' | sort -n | tail -1; }
if [ ! -f zero/cur.bin ]; then cp "$1" zero/cur.bin; cp "$1.pt" zero/cur.pt; echo 0 > zero/prod2.gen; fi
g=$(cat zero/prod2.gen); last=$(top)
python3 zero/farm2.py calc/calc_sp zero/cur.bin $W 100000 >> farm2.log 2>&1 &
say "production started from $1: $W workers, a generation every $STEP shards, window $KEEP shards, $EP passes a generation"
while [ ! -f zero/PROD_STOP ]; do
  sleep 20; now=$(top); [ $(( now - last )) -lt $STEP ] && continue
  last=$now; g=$(( g + 1 ))
  for old in $(ls data/ev_b[0-9]*_c.npy | sed 's/.*ev_b\([0-9]*\)_c.npy/\1/' | sort -n | head -n -$KEEP); do rm -f data/ev_b${old}_c.npy data/ev_b${old}_y.npy; done
  tags=$(ls data/ev_b[0-9]*_c.npy | sed 's/.*ev_\(b[0-9]*\)_c.npy/\1/' | sort -t b -k 2 -n | paste -sd,); t0=$(date +%s)
  python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files $tags --val bval --epochs $EP --lr 1e-3 --threads 4 --device cuda --pst zero/g101s.npz --widen zero/cur.pt --out zero/p$g.bin > zero/p$g.train.log 2>&1
  if grep -q '^done' zero/p$g.train.log; then
    cp zero/p$g.bin zero/cur.bin.tmp && mv zero/cur.bin.tmp zero/cur.bin; cp zero/p$g.bin.pt zero/cur.pt; echo $g > zero/prod2.gen
    say "generation $g: shards up to b$now, $(echo $tags | tr ',' '\n' | wc -l) in the window; played held-out $(grep -h '^done' zero/p$g.train.log | grep -o 'averaged ([0-9.]*' | cut -c11-18); trained in $(( $(date +%s) - t0 ))s"
    ls zero/p*.bin | sed 's/.*\/p\([0-9]*\)\.bin/\1/' | sort -n | head -n -6 | while read o; do [ $(( o % 10 )) -ne 0 ] && rm -f zero/p$o.bin zero/p$o.bin.pt zero/p$o.bin.ema.pt zero/p$o.train.log; done
  else say "generation $g: training failed ($(tail -1 zero/p$g.train.log | cut -c1-120))"; g=$(( g - 1 )); fi
done
touch zero/FARM_STOP; say "production stopped"
