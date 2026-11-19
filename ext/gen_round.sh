#!/bin/bash
# One child network of the generation loop on the outside-data line: the parent's weights trained further, with the
# parent's own recipe, on self-play data of the current player mixed with a slice of all the earlier data; then the child
# against its parent at 30,000 positions per move and against Stockfish 14.1 at 10,000 each.
#   bash ext/gen_round.sh <child> <parent> <width> <engine> <files> [extra trainer options]
# <files>: comma separated  tag:first-last[/step][*repeats]   e.g. "g:0-11*4,u:0-11*2,e:0-11/5,f:0-39/7,h:0-59/9,s:0-406/50"
#          (/step = every step-th shard of the range; *repeats = the shards are listed that many times, i.e. weigh that much more)
# GAMES=<openings> (default 1000 = 2,000 games). The generation farm is frozen (zero/GEN_PAUSE) only while the matches play.
cd /root/loopchess; N=$1; PA=$2; W=$3; E=$4; G=$5; shift 5
say() { echo "[$(date -u +%H:%M:%S)] $*" >> gen_round.log; }
tg=$(python3 - "$G" <<'PY'
import sys
out = []
for part in sys.argv[1].split(','):
    rep = 1
    if '*' in part: part, r = part.split('*'); rep = int(r)
    tag, rng = part.split(':'); step = 1
    if '/' in rng: rng, s = rng.split('/'); step = int(s)
    a, b = (int(x) for x in rng.split('-'))
    out += [f'{tag}{i}' for i in range(a, b + 1, step)] * rep
print(','.join(out))
PY
)
for t in ${tg//,/ }; do [ -e data/ev_${t}_yc.npy ] || ln -s ev_${t}_y.npy data/ev_${t}_yc.npy; done      # self-play labels are already on the networks' scale
O="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; ulimit -n 65536
say "child $N of $PA: $W wide, files $G $*"
t0=$(date +%s)
python3 calc/train_nnue_distill3.py --h $W --buckets 8 --kb 16 --kf 1 --ema 0.999 --files $tg --val ival --epochs 1 --bs 32768 --lr 3e-4 --threads ${THREADS:-8} --device cuda \
  --pst zero/g101s.npz --ytag yc --resume zero/w_$PA.bin.ema.pt --out zero/w_$N.bin "$@" > zero/w_$N.train.log 2>&1
say "network $N trained in $(( $(date +%s) - t0 ))s: $(tail -1 zero/w_$N.train.log | cut -c1-150)"
[ -s zero/w_$N.bin ] || { say "$N: no network written"; exit 1; }
say "engine against trainer: $(python3 zero/check_net.py $E zero/w_$N.bin zero/w_$N.bin.ema.pt 1000 | cut -c1-120)"
while [ -e zero/MATCH_BUSY ]; do sleep 5; done; touch zero/MATCH_BUSY zero/GEN_PAUSE; sleep 2
bash gt.sh "$N against $PA, 30,000 positions per move" gen_round.log 22 --us "$E zero/w_$N.bin" --us_opts "$O" --opp "$PA|$E zero/w_$PA.bin|$O" --openings ${GAMES:-1000} --nodes 30000
bash gt.sh "$N against Stockfish 14.1, 10,000 positions per move each" gen_round.log 22 --us "$E zero/w_$N.bin" --us_opts "$O" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --openings 500 --nodes 10000
rm -f zero/MATCH_BUSY zero/GEN_PAUSE
say "$N finished"
