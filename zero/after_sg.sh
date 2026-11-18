#!/bin/bash
# Jobs that must not compete with clock matches for CPU start here, once the experiments that pause only 48 of the 54
# self-play workers are through (from then on the test queue pauses all 54, which leaves room for these).
cd /root/loopchess
while ! grep -q "\] finished" zero/search_gap.log 2>/dev/null; do sleep 20; done
for p in $(pgrep -f "compact.py archiv[e]"); do kill -CONT $p; done                      # the shard archiver
# the transformer's old training run: stop flag is set, let it save and exit; then the exchange-rate games with 3 workers
for p in $(pgrep -f "mcts:zero/ouro_eva[l]"); do kill -9 $p; done; tmux kill-window -t cont:r16 2>/dev/null
for p in $(pgrep -f "zero/ouro_gp[u].py"); do kill -CONT $p; done
sleep 30
L=zero/research16.log
for pair in 400:2000 400:8000; do
  r=$(OURO_GPU=1 python3 match.py --a mcts:zero/ouro_eval.pt:${pair%%:*}:1:8-2,48-4 --b calc:zero/r16_net.bin:${pair##*:}:calc/calc_w --openings 30 --workers 3 2>/dev/null | tail -1 | cut -c1-400)
  echo "[$(date -u +%H:%M:%S)] transformer ${pair%%:*} evaluations per move vs calculator ${pair##*:} positions per move: $r" >> $L
done
OURO_GPU=1 python3 - >> $L 2>&1 <<'EOP'
import time, torch, chess, random
from engine import Net
net = Net('zero/ouro_eval.pt'); random.seed(1); boards = []
while len(boards) < 1024:
    b = chess.Board()
    for _ in range(random.randint(6, 80)):
        if b.is_game_over(): break
        b.push(random.choice(list(b.legal_moves)))
    if not b.is_game_over(): boards.append(b)
encs, board, castle, ep = net.encode(boards); board, castle, ep = board.to(net.dev), castle.to(net.dev), ep.to(net.dev)
for loops in (1, 2):
    for bs in (16, 256, 1024):
        with torch.no_grad():
            torch.cuda.synchronize(); t = time.time(); n = 0
            while time.time() - t < 2.0:
                ctx = net.model.prepare(board[:bs], castle[:bs], ep[:bs]); s = None
                for _ in range(loops): s = net.model.loop(ctx, s)
                net.model.readout(ctx, s); n += bs
            torch.cuda.synchronize(); dt = time.time() - t
        print(f'network alone on the card: {loops} loop(s), batch {bs}: {n / dt:,.0f} positions per second')
EOP
echo "[$(date -u +%H:%M:%S)] finished" >> $L
