"""One transformer on the graphics card for every board of a match: a small server on a local socket.

  OURO_GPU=1 python3 lm/serve.py <checkpoint> [socket path=/tmp/ouro.sock] [loops=2]

A client (duo_uci.py with OURO_SERVER=<socket path>) sends one line, a FEN, and gets one line back: JSON with the legal
moves, their policy probabilities, the position's value and the value of every move. A line "P <FEN>" (the engine,
calc_s19.c with LmDepth and OURO_SOCK) gets plain text instead: "<move> <log-probability> ..." over the legal moves,
the most likely first (an empty line for a finished game). Requests that arrive within a
millisecond of each other are evaluated as one batch.
"""
import json, math, os, queue, socket, sys, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import chess
from engine import Net

ckpt = sys.argv[1]; path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/ouro.sock'; loops = int(sys.argv[3]) if len(sys.argv) > 3 else 2
net = Net(ckpt, threads=2); Q = queue.Queue()


def batcher():
    while True:
        items = [Q.get()]; t0 = time.time()
        while len(items) < 64 and time.time() - t0 < 0.001:
            try: items.append(Q.get(timeout=0.0005))
            except queue.Empty: pass
        try:
            bs = [chess.Board(f) for f, _ in items]; live = [b for b in bs if any(b.legal_moves)]
            it = iter(net.evaluate(live, loops=loops) if live else [])       # a finished game: no moves, mate 0 or a draw
            res = [next(it) if any(b.legal_moves) else ([], [], 0.0 if b.is_check() else 0.5, []) for b in bs]
            for (f, box), (moves, p, v, q) in zip(items, res):
                if box.plain:
                    o = sorted(zip(p, (m.uci() for m in moves)), reverse=True); box.append(' '.join(f'{u} {math.log(max(float(x), 1e-9)):.3f}' for x, u in o))
                else:
                    box.append(json.dumps({'moves': [m.uci() for m in moves], 'p': [float(x) for x in p], 'v': float(v), 'q': [float(x) for x in q]}))
        except Exception as e:
            for f, box in items: box.append('' if box.plain else json.dumps({'error': str(e)}))
        for f, box in items: box.ev.set()


class Box(list):
    def __init__(self, plain=False): super().__init__(); self.ev = threading.Event(); self.plain = plain


def serve(conn):
    f = conn.makefile('rw')
    for line in f:
        line = line.strip(); plain = line.startswith('P ')
        box = Box(plain); Q.put((line[2:] if plain else line, box)); box.ev.wait(); f.write(box[0] + '\n'); f.flush()


if os.path.exists(path): os.remove(path)
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); srv.bind(path); srv.listen(128)
threading.Thread(target=batcher, daemon=True).start()
net.evaluate([chess.Board()], loops=loops)                    # warm up before the first clock runs
print('ready', path, flush=True)
while True:
    c, _ = srv.accept(); threading.Thread(target=serve, args=(c,), daemon=True).start()
