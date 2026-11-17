"""One transformer on the graphics card for every board of a match: a small server on a local socket.

  OURO_GPU=1 python3 lm/serve.py <checkpoint> [socket path=/tmp/ouro.sock] [loops=2]

A client (duo_uci.py with OURO_SERVER=<socket path>) sends one line, a FEN, and gets one line back: JSON with the legal
moves, their policy probabilities, the position's value and the value of every move. Requests that arrive within a
millisecond of each other are evaluated as one batch.
"""
import json, os, queue, socket, sys, threading, time
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
            res = net.evaluate([chess.Board(f) for f, _ in items], loops=loops)
            for (f, box), (moves, p, v, q) in zip(items, res):
                box.append(json.dumps({'moves': [m.uci() for m in moves], 'p': [float(x) for x in p], 'v': float(v), 'q': [float(x) for x in q]}))
        except Exception as e:
            for f, box in items: box.append(json.dumps({'error': str(e)}))
        for f, box in items: box.ev.set()


class Box(list):
    def __init__(self): super().__init__(); self.ev = threading.Event()


def serve(conn):
    f = conn.makefile('rw')
    for line in f:
        box = Box(); Q.put((line.strip(), box)); box.ev.wait(); f.write(box[0] + '\n'); f.flush()


if os.path.exists(path): os.remove(path)
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); srv.bind(path); srv.listen(128)
threading.Thread(target=batcher, daemon=True).start()
net.evaluate([chess.Board()], loops=loops)                    # warm up before the first clock runs
print('ready', path, flush=True)
while True:
    c, _ = srv.accept(); threading.Thread(target=serve, args=(c,), daemon=True).start()
