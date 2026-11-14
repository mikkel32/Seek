"""Benchmarks on public data: ChessBench test positions (all-move Stockfish Q) and the 10k Lichess puzzles."""
import argparse, csv, json, sys, time
import numpy as np, torch, chess
import torch.nn.functional as F
from engine import Net, MCTS


@torch.no_grad()
def metrics_on(net, boards, meta, moves, qs, loops=2, bs=500, beta=None):
    """Move-quality metrics against all-move reference values (qs = mover's win prob per legal move).
    acc = picked a best move, reg = win-prob given up, bl = blunder rate (gave up > 0.10).
    suffix p = policy head, q = Q head, m = pi x Q mix (what the engine plays searchless)."""
    beta = net.beta if beta is None else beta
    tot = {k: 0.0 for k in ('acc_p', 'acc_q', 'acc_m', 'reg_p', 'reg_q', 'reg_m', 'bl_p', 'bl_q', 'bl_m', 'vmae')}
    n = len(boards)
    for i in range(0, n, bs):
        j = slice(i, i + bs)
        board = torch.from_numpy(boards[j].astype(np.int64))
        castle = torch.from_numpy(meta[j, 0].astype(np.int64)); ep = torch.from_numpy(meta[j, 1].astype(np.int64))
        tm = torch.from_numpy(moves[j].astype(np.int64)); tq = torch.from_numpy(qs[j].astype(np.float32))
        pol, v, ql, _ = net.model(board, castle, ep, loops=loops, tbptl=False)
        valid = tm >= 0; mv = tm.clamp(min=0)
        lp = pol.gather(1, mv).masked_fill(~valid, -1e9)
        lq = ql.gather(1, mv).masked_fill(~valid, -1e9)
        qstar = tq.masked_fill(~valid, -1).max(1).values
        lm = F.log_softmax(lp, 1) + beta * torch.sigmoid(lq)
        for name, sc in (('p', lp), ('q', lq), ('m', lm)):
            r = qstar - tq.gather(1, sc.argmax(1, keepdim=True)).squeeze(1)
            tot['acc_' + name] += (r < 1e-6).sum().item(); tot['reg_' + name] += r.sum().item(); tot['bl_' + name] += (r > 0.1).sum().item()
        tot['vmae'] += (torch.sigmoid(v) - qstar).abs().sum().item()
    return {k: v / n for k, v in tot.items()}


_T = None


def test_metrics(net, half='A', loops=2, n=None, bs=500, beta=None):
    """ChessBench test positions. half A = even indices (held out, reporting only), B = odd (dev)."""
    global _T
    if _T is None:
        z = np.load('data/cb_test.npz')
        _T = {k: z[k] for k in ('boards', 'meta', 'moves', 'qs')}
    idx = np.arange(0 if half == 'A' else 1, len(_T['boards']), 2)
    if n:
        idx = idx[:n]
    return metrics_on(net, _T['boards'][idx], _T['meta'][idx], _T['moves'][idx], _T['qs'][idx], loops, bs, beta)


def load_puzzles(n=None):
    rows = list(csv.DictReader(open('data/puzzles.csv')))
    return rows[:n] if n else rows


def puzzle_accuracy(net, mode='mix', loops=2, n=None, mcts=None):
    """Whole solution must match (ChessBench protocol); any mating move counts on the final move."""
    puz = load_puzzles(n)
    boards = [chess.Board(p['FEN']) for p in puz]
    sols = [p['Moves'].split(' ') for p in puz]
    # the FEN is the position BEFORE the opponent's set-up move: play it, then the solver answers from the second move on
    for bd, sol in zip(boards, sols):
        bd.push(chess.Move.from_uci(sol[0]))
    alive = list(range(len(puz))); pos = [1] * len(puz); ok = [None] * len(puz)
    while alive:
        if mcts is None:
            picks = net.pick([boards[i] for i in alive], mode=mode, loops=loops)
        else:
            picks = [mcts.search(boards[i])[0] for i in alive]
        nxt = []
        for i, mv in zip(alive, picks):
            want = sols[i][pos[i]]
            if mv.uci() != want:
                boards[i].push(mv)
                ok[i] = boards[i].is_checkmate()
                continue
            boards[i].push(mv); pos[i] += 1
            if pos[i] >= len(sols[i]):
                ok[i] = True
                continue
            boards[i].push(chess.Move.from_uci(sols[i][pos[i]])); pos[i] += 1
            if pos[i] >= len(sols[i]):
                ok[i] = True
            else:
                nxt.append(i)
        alive = nxt
    ok = np.array(ok, dtype=bool)
    rating = np.array([int(p['Rating']) for p in puz])
    bands = {}
    for lo in range(400, 3000, 400):
        m = (rating >= lo) & (rating < lo + 400)
        if m.sum():
            bands[f'{lo}-{lo+400}'] = (round(float(ok[m].mean()), 3), int(m.sum()))
    return float(ok.mean()), bands


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('--half', default='A'); ap.add_argument('--n', type=int, default=None)
    ap.add_argument('--loops', default='1,2,4'); ap.add_argument('--puzzles', type=int, default=10000)
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--key', default='ema')
    a = ap.parse_args()
    net = Net(a.ckpt, threads=a.threads, key=a.key)
    out = {'params': net.model.n_params(), 'beta': net.beta}
    for k in [int(x) for x in a.loops.split(',')]:
        t = time.time()
        m = test_metrics(net, a.half, k, a.n)
        out[f'test_k{k}'] = {kk: round(vv, 4) for kk, vv in m.items()}
        if a.puzzles:
            for mode in ('p', 'mix'):
                acc, bands = puzzle_accuracy(net, mode, k, a.puzzles)
                out[f'puzzle_{mode}_k{k}'] = round(acc, 4)
                if mode == 'mix':
                    out[f'puzzle_bands_k{k}'] = bands
        print(f'k={k} done in {time.time()-t:.0f}s', file=sys.stderr, flush=True)
    print(json.dumps(out, indent=1))
