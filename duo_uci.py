"""UCI front end for the two-system engine: CALC searches, OURO adds a prior to the root moves.

  python3 duo_uci.py [calc weights] [ouro checkpoint]

Works in any chess GUI or tournament manager. Options: Prior (centipawns per unit of OURO log-probability,
0 = calculator alone), PriorCap. All `go` parameters are passed through to the calculator.
"""
import subprocess, sys, time
import numpy as np, chess
from engine import Net


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else 'calc/net256k.bin'
    ckpt = sys.argv[2] if len(sys.argv) > 2 else 'ckpt/B.pt'
    calc = subprocess.Popen(['calc/calc', weights], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    ouro = None; lam = 8.0; cap = 40.0
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
            out('option name Hash type spin default 32 min 1 max 4096'); out('uciok')
        elif t[0] == 'isready':
            if ouro is None:
                ouro = Net(ckpt, threads=1)
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
                if ouro is None:
                    ouro = Net(ckpt, threads=1)
                moves, p, v, q = ouro.evaluate([board], loops=2)[0]
                lp = np.log(p + 1e-9); g = np.maximum(lam * (lp - lp.max()), -cap)
                send('bonus ' + ' '.join(f'{m.uci()[:4]} {int(round(x))}' for m, x in zip(moves, g)))
            if 'movetime' in t:                       # OURO's own thinking comes out of the move time
                k = t.index('movetime'); t[k + 1] = str(max(1, int(t[k + 1]) - int(1000 * (time.time() - t0))))
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
