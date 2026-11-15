#!/bin/bash
# One file of Stockfish's own network-training data (Hugging Face, linrock/bullet-training-data) -> our shards -> a plain 384
# network from scratch on them. A separate line: these labels are Stockfish's, the result is NOT self-taught.
#   bash ext/sfdata.sh <path in the repository> <shard prefix, one letter not b or l> <name of the network> [max shards] [passes=6]
cd /root/loopchess; mkdir -p /root/ext; U="https://huggingface.co/datasets/linrock/bullet-training-data/resolve/main/$1"; P=$2; NAME=$3; MAXS=${4:-10000}; EP=${5:-6}
say() { echo "[$(date -u +%H:%M:%S)] $*" >> sfdata.log; }
size=$(curl -sIL "$U" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}'); parts=10; chunk=$(( (size + parts - 1) / parts ))
say "$1: $(( size / 1000000000 )) GB, downloading in $parts pieces"
for i in $(seq 0 $((parts - 1))); do ( for try in 1 2 3 4; do curl -s -L --fail -r $(( i * chunk ))-$(( (i + 1) * chunk - 1 )) "$U" -o /root/ext/sfpart$i && break; sleep 5; done ) & done; wait
got=$(cat /root/ext/sfpart* | wc -c); say "downloaded $(( got / 1000000000 )) GB of $(( size / 1000000000 )); converting"
touch zero/FARM_PAUSE
n=$(for i in $(seq 0 $((parts - 1))); do cat /root/ext/sfpart$i; done | zstd -dc --long=31 2>/dev/null | python3 ext/bullet2shards.py $P 0 $MAXS 2> /root/ext/b2s.log)
rm -f zero/FARM_PAUSE /root/ext/sfpart*
k=$(ls data/ev_${P}[0-9]*_c.npy | wc -l); say "$n positions in $k shards"
tags=$(seq -s, -f "${P}%g" 0 $((k - 1))); t0=$(date +%s)
python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files $tags --val bval --epochs $EP --lr 1e-3 --threads 4 --device cuda --pst zero/g101s.npz --out zero/w_$NAME.bin > zero/w_$NAME.train.log 2>&1
say "network zero/w_$NAME.bin trained from scratch on them, $EP passes, in $(( $(date +%s) - t0 ))s"
say finished
