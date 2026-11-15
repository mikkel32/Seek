#!/bin/bash
# Is the evaluator's size what holds the engine back?
# The standing test showed the engine searches MORE positions per second than Stockfish and still loses heavily, so the
# gap is in judgement. Here five evaluators start from the same production weights, see the same frozen data for the same
# number of epochs, and are then judged where it counts: clock games on equal terms (a bigger evaluator is slower), plus
# games at a fixed number of positions per move (is it smarter per position looked at?).
#   A  384 wide, piece positions          (control: production's shape, simply trained on)
#   B  768 wide                           C  1536 wide
#   D  384 wide, 11 king regions (the input weights depend on where the own king stands)
#   E  384 wide,  7 king regions, mirrored left-right
cd /root/loopchess; L=zero/research5.log; EP=${EP:-4}; : > $L
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
G=${G:-$(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['gen'])")}
SH=${SH:-$(python3 -c "import json;print(' '.join(str(i) for i in json.load(open('zero/cont_state.json'))['shards']))")}
START=zero/r5_start.pt; cp zero/g$G.bin.ema.pt $START; cp zero/g$G.bin zero/r5_S.bin
# freeze the data: hard links cost no disk space and survive the loop dropping a shard from its window
tags=""; for i in $SH; do ln -f data/ev_sh${i}_x.npy data/ev_r5s${i}_x.npy 2>/dev/null && ln -f data/ev_sh${i}_y.npy data/ev_r5s${i}_y.npy 2>/dev/null && tags="$tags,r5s$i"; done; tags=${tags#,}
cp data/ev_shval_x.npy data/ev_r5val_x.npy; cp data/ev_shval_y.npy data/ev_r5val_y.npy
FLAGS="-O3 -march=native -funroll-loops -DNO_PEXT -DHMAX=2048"; gcc $FLAGS -o calc/calc_w calc/calc.c -lm
printf 'position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1\nperft 4\nquit\n' | calc/calc_w | grep -q "perft 4085603" || { say "wide engine build FAILED its self-test"; exit 1; }
say "start: generation $G weights, shards $SH, $EP epochs per arm, wide engine built and verified"
PST=$(ls zero/g${G}s.npz zero/g$((G - 1))s.npz 2>/dev/null | head -1)      # the search's piece values travel inside the network file: the same for every arm
T="python3 calc/train_nnue.py --buckets 8 --ema 0.999 --files $tags --val r5val --epochs $EP --lr 1e-3 --threads 4 --device cuda --pst $PST"
arm() { n=$1; shift; t0=$(date +%s); $T "$@" --out zero/r5_$n.bin > zero/r5_$n.train.log 2>&1
        say "arm $n ($*): $(grep -h '^positions' zero/r5_$n.train.log | cut -c1-110) -> $(grep -h '^done' zero/r5_$n.train.log | tail -1 | cut -c1-140)  [$(( $(date +%s) - t0 ))s]"; }
arm A --h 384  --widen $START
arm B --h 768  --widen $START
arm C --h 1536 --widen $START
arm D --h 384 --kb 11 --warm $START
arm E --h 384 --kb 7  --warm $START
for n in S A B C D E; do
  [ -s zero/r5_$n.bin ] || { say "arm $n has no network"; continue; }
  sp=$(printf 'position startpos\ngo nodes 1500000\nquit\n' | calc/calc_w zero/r5_$n.bin | awk '/^info depth/ {n=$8; t=$10; d=$3} END {printf "%.2fM positions/s, depth %d", n / t / 1000, d}')
  say "arm $n search speed from the opening position: $sp"
done
cat > zero/r5_match.sh <<'EOM'
#!/bin/bash
cd /root/loopchess; L=zero/research5.log
for n in B C D E; do
  [ -s zero/r5_$n.bin ] || continue
  r=$(python3 gauntlet.py --us "calc/calc_w zero/r5_$n.bin" --opp "control A|calc/calc_w zero/r5_A.bin|Hash=64" --tc 10+0.1 --openings 150 --workers 48 --out zero/research5.jsonl | tail -1 | cut -c1-260)
  echo "[$(date -u +%H:%M:%S)] clock 10+0.1, $n vs A: $r" >> $L
  r=$(python3 match.py --a calc:zero/r5_$n.bin:20000:calc/calc_w --b calc:zero/r5_A.bin:20000:calc/calc_w --openings 150 --workers 48 | tail -1 | cut -c1-200)
  echo "[$(date -u +%H:%M:%S)] 20,000 positions per move, $n vs A: $r" >> $L
done
r=$(python3 gauntlet.py --us "calc/calc_w zero/r5_A.bin" --opp "start S|calc/calc_w zero/r5_S.bin|Hash=64" --tc 10+0.1 --openings 150 --workers 48 --out zero/research5.jsonl | tail -1 | cut -c1-260)
echo "[$(date -u +%H:%M:%S)] clock 10+0.1, A vs the start weights: $r" >> $L
EOM
bash zero/burst.sh 48 bash zero/r5_match.sh
say "finished"
