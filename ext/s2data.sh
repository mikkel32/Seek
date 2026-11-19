#!/bin/bash
# Stockfish's second-stage training files (Hugging Face, linrock/bullet-training-data, S2: positions from Leela's self-play
# games with search scores and game results, already filtered to quiet positions) -> our shards s<k>.
# Labels on the Lichess labels' scale (score scale 600, ext/s2cal.py). NOT self-taught, and not Stockfish-played either.
#   bash ext/s2data.sh <iteration 1-12> <first shard number>
cd /root/loopchess; mkdir -p /root/ext; IT=$1; K0=$2
U="https://huggingface.co/datasets/linrock/bullet-training-data/resolve/main/S2/test77nov-unfilt-test79-maraprmay-v6-dd.skip-see-ge0.wdl-pdist.iter-$IT.bullet.bin.zst"
say() { echo "[$(date -u +%H:%M:%S)] $*" >> s2data.log; }
size=$(curl -sIL "$U" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}'); parts=10; chunk=$(( (size + parts - 1) / parts ))
say "iteration $IT: $(( size / 1000000000 )) GB, downloading in $parts pieces"
for i in $(seq 0 $((parts - 1))); do ( for try in 1 2 3 4; do curl -s -L --fail -r $(( i * chunk ))-$(( (i + 1) * chunk - 1 )) "$U" -o /root/ext/s2part$i && break; sleep 5; done ) & done; wait
got=$(cat /root/ext/s2part* | wc -c); say "downloaded $got of $size bytes; converting"
[ "$got" = "$size" ] || { say "download incomplete, stopping"; exit 1; }
n=$(for i in $(seq 0 $((parts - 1))); do cat /root/ext/s2part$i; done | zstd -dc --long=31 2>/dev/null | python3 ext/bullet2shards.py s $K0 10000 600 2> /root/ext/s2conv.log)
rm -f /root/ext/s2part*
say "iteration $IT: $n positions, shards s$K0 to s$(( $(ls data/ev_s[0-9]*_c.npy | wc -l) - 1 )); $(df -h /root | tail -1 | awk '{print $4}') free"
