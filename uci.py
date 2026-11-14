"""UCI front end, so OURO can be plugged into any chess GUI or tournament manager.

  python3 uci.py [checkpoint]

Options: Sims (search simulations per move; 0 = play straight from the network), Loops (searchless only).
`go nodes N` overrides Sims for that move. Clock-based time controls are ignored: strength is set by Sims.
"""
import math, sys
import chess, torch
from engine import Net, MCTS


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else 'ckpt/best.pt'
    net = None; sims = 200; loops = 2; threads = 1
    board = chess.Board(); hist = [board._transposition_key()]
    out = lambda s: (sys.stdout.write(s + '\n'), sys.stdout.flush())
    for line in sys.stdin:
        t = line.split()
        if not t:
            continue
        if t[0] == 'uci':
            out('id name OURO-2M'); out('id author loopchess')
            out('option name Sims type spin default 200 min 0 max 100000')
            out('option name Loops type spin default 2 min 1 max 4')
            out('option name Threads type spin default 1 min 1 max 64')
            out('uciok')
        elif t[0] == 'setoption' and 'name' in t and 'value' in t:
            name = ' '.join(t[t.index('name') + 1:t.index('value')]).lower(); val = int(t[t.index('value') + 1])
            if name == 'sims':
                sims = val
            elif name == 'loops':
                loops = val
            elif name == 'threads':
                threads = val; torch.set_num_threads(threads)
        elif t[0] == 'isready':
            if net is None:
                net = Net(ckpt, threads=threads)
            out('readyok')
        elif t[0] == 'ucinewgame':
            board = chess.Board(); hist = [board._transposition_key()]
        elif t[0] == 'position':
            if t[1] == 'startpos':
                board = chess.Board(); i = 2
            else:
                board = chess.Board(' '.join(t[2:8])); i = 8
            hist = [board._transposition_key()]
            if i < len(t) and t[i] == 'moves':
                for m in t[i + 1:]:
                    board.push_uci(m); hist.append(board._transposition_key())
        elif t[0] == 'go':
            if net is None:
                net = Net(ckpt, threads=threads)
            n = int(t[t.index('nodes') + 1]) if 'nodes' in t else sims
            if n <= 0:
                moves, p, v, q = net.evaluate([board], loops)[0]
                mv = moves[int((torch.tensor(p + 1e-12).log() + net.beta * torch.tensor(q)).argmax())]
                wp = v
            else:
                mv, root = MCTS(net, sims=n, deepen=()).search(board, hist[:-1])
                tot = float(root.N.sum())
                wp = 0.5 * (1.0 + float(root.W.sum()) / tot) if tot > 0 else 0.5 * (1.0 + root.v)
            wp = min(max(wp, 1e-4), 1 - 1e-4)
            out(f'info nodes {max(n, 1)} score cp {int(round(math.log(wp / (1 - wp)) / 0.00368208))} pv {mv.uci()}')
            out('bestmove ' + mv.uci())
        elif t[0] == 'quit':
            break


if __name__ == '__main__':
    main()
