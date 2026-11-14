#!/bin/bash
# Set up a fresh Linux machine and continue the self-taught loop there.
#   tar xzf ouro_zero_bundle.tar.gz && cd loopchess && bash zero/bootstrap.sh
set -e
cd "$(dirname "$0")/.."
export DEBIAN_FRONTEND=noninteractive
(apt-get update -qq && apt-get install -y -qq build-essential stockfish tmux > /dev/null) || echo "apt step failed: install gcc, stockfish and tmux by hand"
python3 -c "import torch" 2>/dev/null || pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q chess numpy
FLAGS="-O3 -march=native -funroll-loops"
grep -q "AuthenticAMD" /proc/cpuinfo && [ "$(awk '/cpu family/ {print $4; exit}' /proc/cpuinfo)" = "23" ] && FLAGS="$FLAGS -DNO_PEXT"     # Zen 1/2: slow PEXT
gcc $FLAGS -o calc/calc_z3 calc/calc.c -lm
printf 'position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1\nperft 4\nquit\n' | calc/calc_z3 | grep -q "perft 4085603" || { echo "engine self-test FAILED"; exit 1; }
echo "engine built and verified"
mkdir -p logs data
NPROC=${NPROC:-$(nproc)}
GAMES_MULT=${GAMES_MULT:-$(python3 -c "print(max(1, min(8, $NPROC // 8)))")}
echo "starting the loop on $NPROC processes, $GAMES_MULT x games per generation; watch with: tail -f zero/loop.log"
START=$(python3 -c "import json; print(max(int(k) for k in json.load(open('zero/rating.json'))))")
tmux new-session -d -s zero "NPROC=$NPROC GAMES_MULT=$GAMES_MULT python3 zero/loop.py $START 14 2>&1 | tee -a zero/loop_stdout.log"
