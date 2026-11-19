#!/bin/bash
# housekeeping once the last match of the label experiment is through
cd /root/loopchess
while ! grep -q "\] finished" zero/research14.log 2>/dev/null; do sleep 10; done
for p in $(pgrep -f "compact.py archiv[e]"); do kill -CONT $p; done                      # the shard archiver
for p in $(pgrep -f "mcts:zero/ouro_eva[l]"); do kill -9 $p; done; tmux kill-window -t cont:r16 2>/dev/null     # the exchange-rate games: answered by the cost measurement
for p in $(pgrep -f "zero/ouro_gp[u].py"); do kill -CONT $p; done                        # the old transformer training: stop flag is set, it saves and exits
sleep 40
OURO_GPU=1 python3 - >> zero/research16.log 2>&1 <<'EOP'
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
for bs in (16, 1024):
    with torch.no_grad():
        torch.cuda.synchronize(); t = time.time(); n = 0
        while time.time() - t < 2.0:
            ctx = net.model.prepare(board[:bs], castle[:bs], ep[:bs]); s = net.model.loop(ctx, None); net.model.readout(ctx, s); n += bs
        torch.cuda.synchronize(); dt = time.time() - t
    print(f'network alone on the card: 1 loop, batch {bs}: {n / dt:,.0f} positions per second')
EOP
echo "[$(date -u +%H:%M:%S)] the remaining games were cancelled (one evaluation costs 20 ms on a CPU thread; the first pairing already put its worth near 14 calculator positions)" >> zero/research16.log
echo "[$(date -u +%H:%M:%S)] finished" >> zero/research16.log
