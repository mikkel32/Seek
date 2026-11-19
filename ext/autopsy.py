"""Where are the games lost?  (game records of gauntlet.py's GAMES_LOG)

  python3 ext/autopsy.py score <games.jsonl> [nodes, default 100000] [workers, default 22]     Stockfish's score of every position -> <games>.sc.jsonl
  python3 ext/autopsy.py report <games.jsonl>

Every position of every game gets a Stockfish score at a fixed node budget (the judge is the opponent's engine at more
nodes than it had in the game; it is not neutral, and its own moves will look good to it). A move's loss is the drop of
the mover's win chance across it. The report: how our losses split into sudden mistakes and slow decline, at which move
numbers and with how much clock, how our per-move loss compares with the opponent's by phase, and how the clock is spent.
"""
import json, os, sys
from multiprocessing import Pool
import chess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quality as Q


def w_score(a):
    nodes, games = a; e = Q.Eng(Q.SF, [('Hash', '64'), ('Threads', '1')]); out = []
    for gi, g in games:
        b = chess.Board(g['fen']); sc = []; e.new()
        for k in range(len(g['moves']) + 1):
            if b.is_game_over():
                sc.append(-30000 if b.is_checkmate() else 0)
            else:
                r = e.lines(b.fen(), nodes); sc.append(r[1][1] if 1 in r else 0)
            if k < len(g['moves']): b.push_uci(g['moves'][k])
        out.append((gi, sc))
    e.quit()
    return out


def score(path, nodes, workers):
    games = [json.loads(l) for l in open(path)]; items = list(enumerate(games)); ch = [items[i:i + 4] for i in range(0, len(items), 4)]; res = {}
    with Pool(workers) as p:
        for r in p.imap_unordered(w_score, [(nodes, c) for c in ch]):
            for gi, sc in r: res[gi] = sc
    with open(path + '.sc.jsonl', 'w') as f:
        for gi, g in enumerate(games):
            g['sc'] = res[gi]; f.write(json.dumps(g) + '\n')
    print(f'{len(games)} games scored at {nodes} nodes a position -> {path}.sc.jsonl')


def report(path):
    games = [json.loads(l) for l in open(path + '.sc.jsonl')]; n = len(games)
    W = sum(g['score'] == 1 for g in games); L = sum(g['score'] == 0 for g in games)
    print(f'{n} games: {W} won, {n - W - L} drawn, {L} lost')
    loss = {'us': {}, 'opp': {}}; cnt = {'us': {}, 'opp': {}}; tm = {}; tmc = {}; left = {}
    lost_kind = {'one move (0.25 or more of a point in a single move of ours)': 0, 'two or three moves (each 0.10 or more)': 0, 'slow decline (no single move of 0.10)': 0}
    when = {}; clock_at = []; big = []
    for g in games:
        b = chess.Board(g['fen']); mover_white = b.turn; us_w = g['us_white']; sc = g['sc']; mv = g['moves']; ms = g['ms']; cl = 10000.0; drops = []
        for k in range(len(mv)):
            ours = (mover_white if k % 2 == 0 else not mover_white) == us_w
            e0 = Q.E(sc[k]); e1 = 1 - Q.E(sc[k + 1]); d = e0 - e1                    # the mover's win chance before and after its move
            mn = (b.fullmove_number if k == 0 else None)
            fm = chess.Board(g['fen']).fullmove_number + (k + (0 if mover_white else 1)) // 2
            ph = 'moves up to 20' if fm <= 20 else 'moves 21-40' if fm <= 40 else 'moves 41-60' if fm <= 60 else 'after move 60'
            side = 'us' if ours else 'opp'; loss[side][ph] = loss[side].get(ph, 0) + max(d, 0); cnt[side][ph] = cnt[side].get(ph, 0) + 1
            if ours:
                tm[ph] = tm.get(ph, 0) + ms[k]; tmc[ph] = tmc.get(ph, 0) + 1; cl = cl - ms[k] + 100
                if fm in (20, 30, 40, 60): left.setdefault(fm, []).append(cl)
                drops.append((d, fm, cl, ms[k], e0))
        if g['score'] == 0:
            d3 = sorted(drops, reverse=True)[:3]; top = d3[0] if d3 else (0, 0, 0, 0, 0)
            kind = list(lost_kind)[0 if top[0] >= 0.25 else 1 if sum(x[0] >= 0.10 for x in d3) >= 1 else 2]; lost_kind[kind] += 1
            # the move from which the game is beyond saving: the last time our win chance (before our move) was 0.35 or more
            lastok = max((x[1] for x in drops if x[4] >= 0.35), default=0); bucket = 'by move 20' if lastok <= 20 else 'moves 21-40' if lastok <= 40 else 'moves 41-60' if lastok <= 60 else 'after move 60'
            when[bucket] = when.get(bucket, 0) + 1
            if top[0] >= 0.10: big.append(top)
    print('\nlost games by how they were lost: ' + '; '.join(f'{k}: {v} ({100 * v / max(1, L):.0f}%)' for k, v in lost_kind.items()))
    print('lost games by the last move at which the judge still gave us 0.35 or more: ' + '; '.join(f'{k}: {when.get(k, 0)}' for k in ('by move 20', 'moves 21-40', 'moves 41-60', 'after move 60')))
    if big:
        big.sort(key=lambda x: x[2]); m = len(big)
        print(f'the worst move of each lost game that had one (0.10 or more; {m} games): median move number {sorted(x[1] for x in big)[m // 2]}, median clock left {sorted(x[2] for x in big)[m // 2] / 1000:.1f}s, '
              f'median time spent on it {sorted(x[3] for x in big)[m // 2]} ms; with under 2 s on the clock: {100 * sum(x[2] < 2000 for x in big) / m:.0f}%, under 1 s: {100 * sum(x[2] < 1000 for x in big) / m:.0f}%')
    print('\nwin chance given away per move (thousandths), ours / the opponent\'s, as the judge sees it:')
    for ph in ('moves up to 20', 'moves 21-40', 'moves 41-60', 'after move 60'):
        if cnt['us'].get(ph):
            print(f'  {ph:15s} {1000 * loss["us"][ph] / cnt["us"][ph]:6.2f} / {1000 * loss["opp"].get(ph, 0) / max(1, cnt["opp"].get(ph, 0)):6.2f}   ({cnt["us"][ph]} of our moves; our time a move {tm[ph] / tmc[ph]:.0f} ms)')
    print('our clock left (10 s to start, 0.1 s a move): ' + '; '.join(f'at move {k}: median {sorted(v)[len(v) // 2] / 1000:.1f}s, a tenth of the games under {sorted(v)[len(v) // 10] / 1000:.1f}s' for k, v in sorted(left.items())))


if __name__ == '__main__':
    if sys.argv[1] == 'score': score(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 100000, int(sys.argv[4]) if len(sys.argv) > 4 else 22)
    else: report(sys.argv[2])
