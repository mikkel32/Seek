"""Read the standing test's results and estimate our rating on the CCRL 40/15 scale from the opponents whose published
ratings are known (single-CPU entries, list of 14 November 2026). Toga II 3.0 has no verified rating and is left out of
the estimate. Draws count as half a point; the rating is the one that makes the observed scores most likely."""
import json, math, sys
ANCHOR = {'Fruit 2.1': 2694, 'Glaurung 2.2': 2793, 'Stockfish 14.1': 3577, 'Stockfish 19': 3629}
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else 'zero/standing.jsonl')]


def fit(rs):
    rs = [r for r in rs if r['opponent'] in ANCHOR]
    if not rs:
        return None
    def ll(R):
        t = 0.0
        for r in rs:
            e = 1 / (1 + 10 ** ((ANCHOR[r['opponent']] - R) / 400)); e = min(max(e, 1e-9), 1 - 1e-9)
            pts = r['w'] + 0.5 * r['d']; t += pts * math.log(e) + (r['n'] - pts) * math.log(1 - e)
        return t
    best = max(range(1500, 3900), key=ll); m = ll(best)
    lo = next(R for R in range(1500, 3900) if ll(R) > m - 1.92); hi = next(R for R in range(3899, 1499, -1) if ll(R) > m - 1.92)
    return best, lo, hi


print(f'{"time control":13s} {"opponent":16s} {"published":>9s}  {"W-D-L":>12s}  {"score":>6s}  {"Elo difference (95% range)":>28s}  time losses us/them')
for r in rows:
    print(f'{r["tc"]:13s} {r["opponent"]:16s} {str(ANCHOR.get(r["opponent"], "-")):>9s}  {str(r["w"]) + "-" + str(r["d"]) + "-" + str(r["l"]):>12s}  {100 * r["score"]:5.1f}%  '
          f'{r["elo"]:+7.0f} ({r["lo"]:+.0f} to {r["hi"]:+.0f})'.ljust(30) + f'   {r["our_time_losses"]}/{r["their_time_losses"]}' + (f'  illegal {r["illegal"]}' if r.get('illegal') else ''))
for tc in sorted({r['tc'] for r in rows}, key=lambda t: float(t.split('+')[0])) + ['all']:
    f = fit([r for r in rows if tc == 'all' or r['tc'] == tc])
    if f:
        print(f'estimated rating on the CCRL 40/15 scale, {tc if tc != "all" else "all time controls together"}: {f[0]} (95% range {f[1]} to {f[2]})')
