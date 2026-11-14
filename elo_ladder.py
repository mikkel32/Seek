"""Rating on Stockfish 16's UCI_Elo scale: play a ladder of strength-limited Stockfish levels, fit one rating.

  python3 elo_ladder.py --player mcts:ckpt/B.pt:64:1:none --start 1700 --openings 20 --levels 4 --out logs/elo_x.json

Each level: `openings` balanced openings, each played twice with colours swapped. The next level moves
towards the 50% point. The rating is the maximum-likelihood value under the logistic Elo model across
all levels (draw = half a point); the interval uses the observed per-game score variance.
"""
import argparse, json, math, time
from match import play

LO, HI = 1320, 3190          # Stockfish 16's UCI_Elo range


def fit(rows):
    """rows: (opp_elo, w, d, l) -> rating, 95% half-width (None when every game went the same way)"""
    pts = sum(w + 0.5 * d for _, w, d, l in rows); n = sum(w + d + l for _, w, d, l in rows)
    if pts <= 0 or pts >= n:
        return None, None
    exp = lambda R: sum((w + d + l) / (1 + 10 ** ((e - R) / 400)) for e, w, d, l in rows)
    a, b = 0.0, 5000.0
    for _ in range(60):
        m = (a + b) / 2
        a, b = (m, b) if exp(m) < pts else (a, m)
    R = (a + b) / 2
    num = 0.0; den = 0.0
    for e, w, d, l in rows:
        k = w + d + l; s = (w + 0.5 * d) / k; p = 1 / (1 + 10 ** ((e - R) / 400))
        num += w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s ** 2          # sum of per-game variances
        den += k * p * (1 - p) * math.log(10) / 400
    return R, 1.96 * math.sqrt(num) / den


def next_level(done):
    below = [e for e, s in done.items() if s >= 0.5]; above = [e for e, s in done.items() if s < 0.5]
    if not above:
        nxt = max(below) + 200
    elif not below:
        nxt = min(above) - 200
    else:
        lo, hi = max(below), min(above)
        if hi - lo <= 100:
            return None
        nxt = int(round((lo + hi) / 2 / 50.0)) * 50
    nxt = min(HI, max(LO, nxt))
    return None if nxt in done else nxt


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--player', required=True); ap.add_argument('--start', type=int, default=1700)
    ap.add_argument('--openings', type=int, default=20); ap.add_argument('--levels', type=int, default=4)
    ap.add_argument('--sf_time', type=float, default=0.1); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    rows = []; done = {}; level = a.start; log = []
    while level is not None and len(rows) < a.levels:
        t = time.time()
        r = play(a.player, f'sf:{level}:{a.sf_time}', a.openings, a.workers, seed=level)
        rows.append((level, r['w'], r['d'], r['l'])); done[level] = r['score']
        R, ci = fit(rows)
        r.update(level=level, secs=round(time.time() - t), rating_so_far=None if R is None else round(R), ci=None if ci is None else round(ci))
        log.append(r); print(json.dumps(r), flush=True)
        level = next_level(done)
    R, ci = fit(rows)
    out = dict(player=a.player, sf_time=a.sf_time, rating=None if R is None else round(R), ci95=None if ci is None else round(ci),
               games=sum(w + d + l for _, w, d, l in rows), levels=log)
    print(json.dumps(dict(player=a.player, rating=out['rating'], ci95=out['ci95'], games=out['games'])), flush=True)
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)
