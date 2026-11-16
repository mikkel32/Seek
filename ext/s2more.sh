#!/bin/bash
# More of Stockfish's second-stage files (see ext/s2data.sh), each turned straight into the labels the full-scale runs use
# (_yc.npy, ext/s2yc.py) with the intermediate label files removed. Room is made first by dropping the Lichess game-position
# shards h60-h209 (2.56 billion positions; at probe scale the share of Lichess positions in the mix does not matter, and
# they can be rebuilt with ext/fishnet.sh) and the score-only labels of the engine-game shards already converted.
#   bash ext/s2more.sh <iteration> ...
cd /root/loopchess; mkdir -p /root/ext
say() { echo "[$(date -u +%H:%M:%S)] $*" >> s2data.log; }
for k in $(seq 60 209); do rm -f data/ev_h${k}_c.npy data/ev_h${k}_y.npy data/ev_h${k}_ym.npy data/ev_h${k}_yc.npy; done
for f in data/ev_s[0-9]*_yc.npy; do rm -f ${f%_yc.npy}_y.npy ${f%_yc.npy}_ym.npy; done
say "room made: $(df -h /root | tail -1 | awk '{print $4}') free"
for IT in "$@"; do
  K0=$(ls data/ev_s[0-9]*_c.npy | wc -l)
  U="https://huggingface.co/datasets/linrock/bullet-training-data/resolve/main/S2/test77nov-unfilt-test79-maraprmay-v6-dd.skip-see-ge0.wdl-pdist.iter-$IT.bullet.bin.zst"
  size=$(curl -sIL "$U" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}'); parts=10; chunk=$(( (size + parts - 1) / parts ))
  for i in $(seq 0 $((parts - 1))); do ( for try in 1 2 3 4; do curl -s -L --fail -r $(( i * chunk ))-$(( (i + 1) * chunk - 1 )) "$U" -o /root/ext/s2part$i && break; sleep 5; done ) & done; wait
  got=$(cat /root/ext/s2part* | wc -c); [ "$got" = "$size" ] || { say "iteration $IT: download incomplete ($got of $size), stopping"; rm -f /root/ext/s2part*; exit 1; }
  n=$(for i in $(seq 0 $((parts - 1))); do cat /root/ext/s2part$i; done | zstd -dc --long=31 2>/dev/null | python3 ext/bullet2shards.py s $K0 10000 600 2> /root/ext/s2conv.log)
  rm -f /root/ext/s2part*; K1=$(( $(ls data/ev_s[0-9]*_c.npy | wc -l) - 1 ))
  python3 ext/s2yc.py $K0 $K1 > /root/ext/s2yc.log 2>&1
  for k in $(seq $K0 $K1); do [ -s data/ev_s${k}_yc.npy ] && rm -f data/ev_s${k}_y.npy data/ev_s${k}_ym.npy; done
  say "iteration $IT: $n positions, shards s$K0 to s$K1, labels ready; $(df -h /root | tail -1 | awk '{print $4}') free"
done
