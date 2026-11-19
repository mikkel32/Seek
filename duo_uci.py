"""UCI front end for the two-system engine: CALC searches, OURO adds a prior to the root moves.

  python3 duo_uci.py [calc weights] [ouro checkpoint]

Works in any chess GUI or tournament manager. Options: Prior (centipawns per unit of OURO log-probability,
0 = calculator alone), PriorCap. All `go` parameters are passed through to the calculator.

TwoLines=1 (with `go nodes N` only, for the quality test): the calculator searches LineSplit% of the nodes and reports its
best move a and line; a is then held back by 30 pawns and the rest of the nodes find the runner-up b and its line (the
hash table carries over). The transformer values the position at the end of each line and the move is chosen by
  (1 - Leaf/100) x search score / 210  +  Leaf/100 x 1.25 x logit(transformer's win chance at the line's end)
for the root's mover (both in logit units of win chance, each at the scale that fits Stockfish's deep scores best:
ext/brain_metric.py, ext/brain_lm.py). Leaf=0 is the control: the same two searches, chosen by score alone.
"""
import json, os, socket, subprocess, sys, time
import numpy as np, chess


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else 'calc/net256k.bin'
    ckpt = sys.argv[2] if len(sys.argv) > 2 else 'ckpt/B.pt'
    calc = subprocess.Popen([os.environ.get('CALC_BIN', 'calc/calc'), weights], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)   # CALC_BIN: which calculator binary
    ouro = None; srv = srvf = None; lam = 8.0; cap = 40.0; two = 0; leaf = 50.0; split = 50.0
    board = chess.Board()
    out = lambda s: (sys.stdout.write(s + '\n'), sys.stdout.flush())
    send = lambda s: (calc.stdin.write(s + '\n'), calc.stdin.flush())
    for line in sys.stdin:
        line = line.strip(); t = line.split()
        if not t:
            continue
        if t[0] == 'uci':
            out('id name OURO+CALC'); out('id author loopchess')
            out('option name Prior type spin default 8 min 0 max 100'); out('option name PriorCap type spin default 40 min 0 max 400')
            out('option name TwoLines type spin default 0 min 0 max 1'); out('option name Leaf type spin default 50 min 0 max 100'); out('option name LineSplit type spin default 50 min 10 max 90')
            out('option name Hash type spin default 32 min 1 max 4096'); out('uciok')
        elif t[0] == 'isready':
            if ouro is None and not os.environ.get('OURO_SERVER'):
                ouro = __import__('engine').Net(ckpt, threads=1)
            send('isready')
            while calc.stdout.readline().strip() != 'readyok':
                pass
            out('readyok')
        elif t[0] == 'setoption' and 'value' in t:
            name = ' '.join(t[t.index('name') + 1:t.index('value')]).lower(); val = t[t.index('value') + 1]
            if name == 'prior':
                lam = float(val)
            elif name == 'priorcap':
                cap = float(val)
            elif name == 'twolines':
                two = int(val)
            elif name == 'leaf':
                leaf = float(val)
            elif name == 'linesplit':
                split = float(val)
            else:
                send(line)
        elif t[0] == 'position':
            send(line)
            if t[1] == 'startpos':
                board = chess.Board(); i = 2
            else:
                board = chess.Board(' '.join(t[2:8])); i = 8
            if i < len(t) and t[i] == 'moves':
                for m in t[i + 1:]:
                    board.push_uci(m)
        elif t[0] == 'go':
            t0 = time.time()
            if lam > 0 and not board.is_game_over():
                if os.environ.get('OURO_SERVER'):      # the transformer runs once, on the graphics card, for every board (lm/serve.py)
                    if srv is None:
                        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); srv.connect(os.environ['OURO_SERVER']); srvf = srv.makefile('rw')
                    srvf.write(board.fen() + '\n'); srvf.flush(); r = json.loads(srvf.readline())
                    moves, p = [chess.Move.from_uci(u) for u in r['moves']], np.array(r['p'])
                else:
                    if ouro is None:
                        ouro = __import__('engine').Net(ckpt, threads=1)
                    moves, p, v, q = ouro.evaluate([board], loops=2)[0]
                lp = np.log(p + 1e-9); g = np.maximum(lam * (lp - lp.max()), -cap)
                send('bonus ' + ' '.join(f'{m.uci()[:4]} {int(round(x))}' for m, x in zip(moves, g)))
            if 'movetime' in t:                       # OURO's own thinking comes out of the move time
                k = t.index('movetime'); t[k + 1] = str(max(1, int(t[k + 1]) - int(1000 * (time.time() - t0))))
            ck = 'wtime' if board.turn == chess.WHITE else 'btime'      # ... and out of the mover's clock
            if ck in t:
                k = t.index(ck); t[k + 1] = str(max(1, int(t[k + 1]) - int(1000 * (time.time() - t0))))
            if two and 'nodes' in t and not board.is_game_over():
                if srv is None and os.environ.get('OURO_SERVER'):
                    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); srv.connect(os.environ['OURO_SERVER']); srvf = srv.makefile('rw')
                k = t.index('nodes'); n = int(t[k + 1]); n1 = max(1, int(n * split / 100)); lines = []
                for part, extra in ((n1, ''), (n - n1, None)):
                    if extra is None:
                        if len(list(board.legal_moves)) < 2:
                            break
                        send(f'bonus {lines[0][0]} -3000')
                    t[k + 1] = str(part); send(' '.join(t)); sc = None; pv = []
                    while True:
                        r = calc.stdout.readline()
                        if not r:
                            return
                        r = r.strip()
                        if r.startswith('info') and ' pv ' in r and ' score ' in r:
                            w = r.split(); j = w.index('score')
                            sc = (int(w[j + 2]) if w[j + 1] == 'cp' else (100000 if int(w[j + 2]) > 0 else -100000)); pv = w[w.index('pv') + 1:]
                        if r.startswith('bestmove'):
                            bm = r.split()[1]; break
                    if not pv or pv[0] != bm:
                        pv = [bm]
                    if lines and bm == lines[0][0]:
                        break
                    lines.append((bm, sc if sc is not None else 0, pv))
                best = lines[0][0]
                if len(lines) == 2 and srvf is not None and max(abs(x[1]) for x in lines) < 100000:
                    zs = []
                    for bm, sc, pv in lines:
                        b = board.copy(stack=False); flip = False
                        for u in pv:
                            mv = chess.Move.from_uci(u)
                            if mv not in b.legal_moves:
                                break
                            b.push(mv); flip = not flip
                        if b.is_game_over():
                            v = 0.0 if b.is_checkmate() else 0.5
                        else:
                            srvf.write(b.fen() + '\n'); srvf.flush(); v = float(json.loads(srvf.readline())['v'])
                        v = 1 - v if flip else v; v = min(max(v, 1e-4), 1 - 1e-4)
                        zs.append((1 - leaf / 100) * sc / 210 + (leaf / 100) * 1.25 * float(np.log(v / (1 - v))))
                    best = lines[1][0] if zs[1] > zs[0] else lines[0][0]
                elif len(lines) == 2 and lines[1][1] > lines[0][1]:
                    best = lines[1][0]
                out(f'bestmove {best}')
                continue
            send(' '.join(t))
            while True:
                r = calc.stdout.readline()
                if not r:
                    return
                out(r.strip())
                if r.startswith('bestmove'):
                    break
        elif t[0] == 'quit':
            send('quit'); break
        else:
            send(line)


if __name__ == '__main__':
    main()
