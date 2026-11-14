#!/bin/bash
# Do eight narrow phase specialists match one wide network at the cost of a narrow one?
#   A  one network, 128 hidden units            (today's design)
#   B  eight phase specialists, 64 units each   (each position touches only its own specialist)
#   C  one network, 64 hidden units             (same cost as B)
cd "$(dirname "$0")/.."
rm -f zero/p[abct].bin*
D="--buckets 8 --ema 0.999 --files z3t,z4t --val z4v --epochs 2 --lr 1e-3 --pst zero/g5s.npz --seed 3 --threads 3"
T() { python3 calc/train_nnue.py "$@" 2>&1 | grep -E "^done" | sed 's/.*averaged (\([0-9.]*\), \([0-9.]*\)).*/\1/'; }
echo "validation log-loss A (128 units):          $(T --h 128 $D --out zero/pa.bin)"
echo "validation log-loss B (8 specialists x 64): $(T --h 64 --kb 8 --phase $D --out zero/pb.bin)"
echo "validation log-loss C (64 units):           $(T --h 64 $D --out zero/pc.bin)"
python3 calc/check_net.py zero/pb.bin.ema.pt zero/pb.bin calc/calc_x
# training cost: CPU seconds for 2M positions on one thread, minus the same run on 0.2M positions (start-up and validation)
for cfg in "--h 128" "--h 64 --kb 8 --phase" "--h 64"; do
  python3 - "$cfg" <<'PY'
import resource, subprocess, sys
cfg = sys.argv[1].split()
base = ['python3', 'calc/train_nnue.py'] + cfg + '--buckets 8 --files z4t --val z4v --epochs 1 --pst zero/g5s.npz --threads 1 --out zero/pt.bin'.split()
def cpu(rows):
    t0 = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
    subprocess.run(base + ['--max_rows', str(rows)], capture_output=True)
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime - t0
big, small = cpu(2_000_000), cpu(200_000)
print('training cost [%s]: %.0f positions per CPU-second' % (' '.join(cfg), 1.8e6 / max(0.1, big - small)))
PY
done
# search speed: nodes per second on the same position
for n in pa pb pc; do
  printf 'position startpos moves e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6\ngo nodes 3000000\nquit\n' | calc/calc_x zero/$n.bin | grep "^info depth" | tail -1 | awk -v n=$n '{for (i = 1; i <= NF; i++) { if ($i == "nodes") N = $(i + 1); if ($i == "time") T = $(i + 1) } printf "search speed %s: %.0f thousand nodes per second\n", n, N / T}'
done
python3 match.py --a calc:zero/pb.bin:5000:calc/calc_x --b calc:zero/pa.bin:5000:calc/calc_x --openings 200 --workers 2 2>/dev/null | tail -1 | python3 -c "import sys, json; d = json.loads(sys.stdin.read()); print('B vs A, equal nodes:', d['w'], d['d'], d['l'], 'score', d['score'], 'Elo', d['elo'], (d['lo'], d['hi']))"
python3 match.py --a calc:zero/pb.bin:0.05:calc/calc_x --b calc:zero/pa.bin:0.05:calc/calc_x --openings 200 --workers 2 2>/dev/null | tail -1 | python3 -c "import sys, json; d = json.loads(sys.stdin.read()); print('B vs A, equal time :', d['w'], d['d'], d['l'], 'score', d['score'], 'Elo', d['elo'], (d['lo'], d['hi']))"
python3 match.py --a calc:zero/pb.bin:0.05:calc/calc_x --b calc:zero/pc.bin:0.05:calc/calc_x --openings 200 --workers 2 2>/dev/null | tail -1 | python3 -c "import sys, json; d = json.loads(sys.stdin.read()); print('B vs C, equal time :', d['w'], d['d'], d['l'], 'score', d['score'], 'Elo', d['elo'], (d['lo'], d['hi']))"
