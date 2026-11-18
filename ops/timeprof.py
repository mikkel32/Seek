import json, sys, chess
games = [json.loads(l) for l in open(sys.argv[1])]
bands = [(1, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 40), (41, 60), (61, 400)]
t = {s: {b: [0, 0] for b in bands} for s in ('us', 'opp')}; first = []
for g in games:
    b = chess.Board(g['fen']); w = b.turn; fm0 = b.fullmove_number; first.append(fm0)
    for k, ms in enumerate(g['ms']):
        fm = fm0 + (k + (0 if w else 1)) // 2; side = 'us' if ((w if k % 2 == 0 else not w) == g['us_white']) else 'opp'
        for bd in bands:
            if bd[0] <= fm <= bd[1]: t[side][bd][0] += ms; t[side][bd][1] += 1
print(f'games start at move {min(first)} to {max(first)} (median {sorted(first)[len(first) // 2]})')
print('move numbers: our ms a move / the opponent\'s ms a move (moves counted)')
for bd in bands:
    u, o = t['us'][bd], t['opp'][bd]
    if u[1]: print(f'  {bd[0]:3d}-{bd[1]:3d}: {u[0] / u[1]:6.0f} / {o[0] / max(1, o[1]):6.0f}   ({u[1]})')
tot = {s: sum(v[0] for v in t[s].values()) / len(games) / 1000 for s in t}
print(f'time used a game: ours {tot["us"]:.1f}s, the opponent\'s {tot["opp"]:.1f}s')
