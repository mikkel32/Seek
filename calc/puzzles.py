"""Puzzle accuracy of a UCI engine, ChessBench protocol: the whole solution must match (any mate counts on the last move)."""
import sys, csv, chess, chess.engine
from multiprocessing import Pool

def work(args):
    cmd, t, rows = args
    e = chess.engine.SimpleEngine.popen_uci(cmd); ok = []
    for fen, moves, rating in rows:
        b = chess.Board(fen); sol = moves.split(' '); good = True
        b.push(chess.Move.from_uci(sol[0]))              # the opponent's set-up move; the solver starts from the second move
        for i in range(1, len(sol), 2):
            m = e.play(b, chess.engine.Limit(time=t)).move
            if m.uci() != sol[i]:
                b.push(m); good = b.is_checkmate(); break
            b.push(m)
            if i + 1 < len(sol): b.push(chess.Move.from_uci(sol[i + 1]))
        ok.append((good, rating))
    e.quit()
    return ok

if __name__ == '__main__':
    cmd = sys.argv[1].split(); t = float(sys.argv[2]); n = int(sys.argv[3])
    rows = [(r['FEN'], r['Moves'], int(r['Rating'])) for r in csv.DictReader(open('data/puzzles.csv'))][:n]
    with Pool(4) as p:
        res = [x for part in p.map(work, [(cmd, t, rows[i::4]) for i in range(4)]) for x in part]
    print(f'puzzles {len(res)} solved {sum(g for g, _ in res) / len(res):.4f}', end=' | ')
    for lo in range(400, 3000, 400):
        sel = [g for g, r in res if lo <= r < lo + 400]
        if sel: print(f'{lo}-{lo+400}: {sum(sel)/len(sel):.2f} ({len(sel)})', end='  ')
    print()
