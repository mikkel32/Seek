#!/bin/bash
# The first part of one of Stockfish's first-stage training files (Hugging Face, linrock/bullet-training-data, S1: positions
# from Stockfish's own self-play from unbalanced openings, a 5,000-node search score and the game result) -> our shards t<k>,
# streamed (nothing but the shards touches the disk). Score scale 950 = Stockfish's internal units on the Lichess labels'
# scale; _y.npy holds the score alone, _ym.npy 0.75 score + 0.25 result. An earlier test of this data (network SF1) used
# scale 280, which inflated every label about 3.4 times and is void.
#   bash ext/s1data.sh <iteration 1-10> <first shard number> <shards to make>
cd /root/loopchess; IT=$1; K0=$2; MAXS=$3
U="https://huggingface.co/datasets/linrock/bullet-training-data/resolve/main/S1/UHO.pdist.iter-$IT.bullet.bin.zst"
n=$(curl -s -L --fail "$U" | zstd -dc --long=31 2>/dev/null | python3 ext/bullet2shards.py t $K0 $MAXS 950 2> /root/ext/s1conv.log)
echo "[$(date -u +%H:%M:%S)] first-stage file $IT: $n positions, shards t$K0 to t$(( K0 + MAXS - 1 )); $(df -h /root | tail -1 | awk '{print $4}') free" >> s2data.log
