"""One transformer on the graphics card for every board of a match: a small server on a local socket.

  OURO_GPU=1 python3 lm/serve.py <checkpoint> [socket path=/tmp/ouro.sock] [loops=2]

Three kinds of request, one line each:
  <FEN>          (duo_uci.py with OURO_SERVER) -> JSON: the legal moves, their policy probabilities, the position's value
                 and the value of every move
  P <FEN>        -> "<move> <log-probability> ..." over the legal moves, the most likely first (empty for a finished game)
  Q <id> <64 piece codes 0-c> <castling> <ep> <n> <move index>...
                 (the engine, calc_s19.c with LmDepth and OURO_SOCK) the board already in the transformer's frame
                 (chessenc.py) and the moves to score -> "<id> <policy logit x100, integer>..." in the same order. No chess
                 work on this side; the engine does not wait for the answer unless LmWait=1.

One thread does everything around a selector: it reads whatever has arrived from every client, evaluates all waiting
encoded requests as one batch (up to 1024), and writes the answers back without blocking. (A thread per client was
limited to ~2,000 answers a second by the interpreter lock; engines ask ~20,000 a second in a 22-board test.)
Encoded requests older than OURO_STALE seconds (default 0.1) are dropped: the search that asked has moved on.
kill -USR1 <pid> writes every thread's stack to the log.
"""
import faulthandler, json, math, os, resource, selectors, signal, socket, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import chess
from engine import Net

ckpt = sys.argv[1]; path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/ouro.sock'; loops = int(sys.argv[3]) if len(sys.argv) > 3 else 2
net = Net(ckpt, threads=2)
STALE = float(os.environ.get('OURO_STALE', '0.1')); MAXB = 1024
STATS = {'in': 0, 'answered': 0, 'stale': 0, 'batches': 0}
HEX = np.zeros(256, np.int64); HEX[np.frombuffer(b'0123456789abc', np.uint8)] = np.arange(13)
faulthandler.register(signal.SIGUSR1, all_threads=True)
resource.setrlimit(resource.RLIMIT_NOFILE, (resource.getrlimit(resource.RLIMIT_NOFILE)[1],) * 2)


class Client:
    def __init__(self, sock): self.sock = sock; self.inb = b''; self.outb = bytearray(); self.alive = True


@torch.no_grad()
def encoded(reqs):
    """[(client, line without 'Q ')] -> the answers are appended to each client's output"""
    B = len(reqs); ws = [r[1].split() for r in reqs]
    bd = HEX[np.frombuffer(b''.join(w[1] for w in ws), np.uint8)].reshape(B, 64)
    ca = np.array([int(w[2]) for w in ws], np.int64); ep = np.array([int(w[3]) for w in ws], np.int64)
    mv = [np.array(w[5:5 + int(w[4])], np.int64) for w in ws]
    dev = net.dev; m = net.model
    ctx = m.prepare(torch.from_numpy(bd).to(dev), torch.from_numpy(ca).to(dev), torch.from_numpy(ep).to(dev)); st = None
    for _ in range(loops): st = m.loop(ctx, st)
    pol = m.readout(ctx, st)[0].float().reshape(B, -1)
    # gather on the card: one row of move indices per request, padded
    L = max(1, max(len(x) for x in mv)); idx = np.zeros((B, L), np.int64)
    for i, x in enumerate(mv): idx[i, :len(x)] = x
    g = torch.round(pol.gather(1, torch.from_numpy(idx).to(dev)) * 100).to(torch.int32).cpu().numpy()
    for i, (c, line) in enumerate(reqs):
        c.outb += ws[i][0] + b' ' + b' '.join(b'%d' % x for x in g[i, :len(mv[i])]) + b'\n'


def fen_reply(kind, fens):
    bs = [chess.Board(f) for f in fens]; live = [b for b in bs if any(b.legal_moves)]
    it = iter(net.evaluate(live, loops=loops) if live else [])                 # a finished game: no moves, mate 0 or a draw
    out = []
    for b in bs:
        moves, p, v, q = next(it) if any(b.legal_moves) else ([], [], 0.0 if b.is_check() else 0.5, [])
        if kind == 'P':
            o = sorted(zip(p, (m.uci() for m in moves)), reverse=True); out.append(' '.join(f'{u} {math.log(max(float(x), 1e-9)):.3f}' for x, u in o))
        else:
            out.append(json.dumps({'moves': [m.uci() for m in moves], 'p': [float(x) for x in p], 'v': float(v), 'q': [float(x) for x in q]}))
    return out


if os.path.exists(path): os.remove(path)
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); srv.bind(path); srv.listen(512); srv.setblocking(False)
sel = selectors.DefaultSelector(); sel.register(srv, selectors.EVENT_READ, None)
net.evaluate([chess.Board()], loops=loops)                    # warm up before the first clock runs
print('ready', path, flush=True)
pend = []                                                     # encoded requests waiting: (arrival time, client, line)
last = time.time(); last_stats = dict(STATS)


def close(c):
    c.alive = False
    try: sel.unregister(c.sock)
    except (KeyError, ValueError): pass
    c.sock.close()


def flush(c):
    if not c.alive or not c.outb: return
    try:
        n = c.sock.send(c.outb); del c.outb[:n]
    except BlockingIOError:
        pass
    except OSError:
        close(c); return
    sel.modify(c.sock, selectors.EVENT_READ | (selectors.EVENT_WRITE if c.outb else 0), c)


while True:
    try:
        for key, ev in sel.select(timeout=0 if pend else 0.5):
            if key.data is None:                              # a new client
                try:
                    s, _ = srv.accept(); s.setblocking(False); sel.register(s, selectors.EVENT_READ, Client(s))
                except BlockingIOError:
                    pass
                continue
            c = key.data
            if ev & selectors.EVENT_WRITE: flush(c)
            if not (ev & selectors.EVENT_READ) or not c.alive: continue
            try:
                d = c.sock.recv(1 << 20)
            except BlockingIOError:
                continue
            except OSError:
                d = b''
            if not d:
                close(c); continue
            c.inb += d; *lines, c.inb = c.inb.split(b'\n'); now = time.time(); slow = []
            for l in lines:
                l = l.strip()
                if not l: continue
                STATS['in'] += 1
                if l.startswith(b'Q '): pend.append((now, c, l[2:]))
                else: slow.append(l)
            for l in slow:                                    # plain and JSON requests: the client waits, answer at once
                kind = 'P' if l.startswith(b'P ') else 'J'
                c.outb += (fen_reply(kind, [l[2:].decode() if kind == 'P' else l.decode()])[0] + '\n').encode(); flush(c)
        if pend:
            now = time.time(); fresh = [(c, l) for t, c, l in pend if c.alive and now - t <= STALE]
            STATS['stale'] += len(pend) - len(fresh); pend = []
            for i in range(0, len(fresh), MAXB):
                encoded(fresh[i:i + MAXB]); STATS['batches'] += 1; STATS['answered'] += len(fresh[i:i + MAXB])
            for c in {id(c): c for c, _ in fresh}.values(): flush(c)
        if time.time() - last > 30:                           # throughput, every 30 seconds
            d = {k: STATS[k] - last_stats[k] for k in STATS}; dt = time.time() - last; last = time.time(); last_stats = dict(STATS)
            if d['in']: print(f"[{time.strftime('%H:%M:%S')}] per second: {d['in'] / dt:.0f} requests, {d['answered'] / dt:.0f} answered, {d['stale'] / dt:.0f} dropped as stale; {d['answered'] / max(1, d['batches']):.0f} a batch", flush=True)
    except Exception as e:                                    # one bad request or batch must never stop the server
        print('server error:', repr(e), flush=True); pend = []
