#!/bin/bash
# What is one evaluation by the transformer worth inside a search, in positions searched by the calculator?
# The transformer (2M parameters) judges a position about as well as the calculator does after searching 2,500 positions,
# but it costs thousands of times more per judgement on a CPU. On the graphics card it can judge tens of thousands of
# positions a second in batches. Whether a search built on it could ever out-play the calculator depends on one number:
# how many calculator positions one transformer evaluation replaces in actual play. Measured here by games in which
# neither side has a clock: the transformer's tree search gets a fixed number of evaluations per move, the calculator a
# fixed number of positions per move, 100 games per pairing (both colours of 50 openings).
cd /root/loopchess; L=zero/research16.log; : > $L
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
cp zero/ouro.pt zero/ouro_eval.pt; G=$(python3 -c "import json;print(json.load(open('zero/cont_state.json'))['gen'])"); cp zero/g$G.bin zero/r16_net.bin
say "transformer at $(tail -1 zero/ouro.log | grep -o 'step [0-9]*'), calculator generation $G"
pkill -STOP -f zero/ouro_gpu.py                      # its training would share the card
m() { r=$(OURO_GPU=1 python3 match.py --a mcts:zero/ouro_eval.pt:$1:1:8-2,48-4 --b calc:zero/r16_net.bin:$2:calc/calc_w --openings ${3:-50} --workers ${W:-6} 2>/dev/null | tail -1 | cut -c1-400)
      say "transformer $1 evaluations per move vs calculator $2 positions per move: $r"; }
for pair in ${PAIRS:-100:500 100:2000 400:2000 400:8000}; do m ${pair%%:*} ${pair##*:} ${OPENINGS:-30}; done
pkill -CONT -f zero/ouro_gpu.py
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
for loops in (1, 2, 4):
    for bs in (16, 256, 1024):
        with torch.no_grad():
            for rep in range(2):
                torch.cuda.synchronize(); t = time.time(); n = 0
                while time.time() - t < 1.5:
                    ctx = net.model.prepare(board[:bs], castle[:bs], ep[:bs]); s = None
                    for _ in range(loops): s = net.model.loop(ctx, s)
                    net.model.readout(ctx, s); n += bs
                torch.cuda.synchronize(); dt = time.time() - t
        print(f'network alone on the card: {loops} loop(s), batch {bs}: {n / dt:,.0f} positions per second')
EOP
say "finished"
