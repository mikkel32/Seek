import json, sys, chess
games = [json.loads(l) for l in open(sys.argv[1])]
bands = [(1, 8), (9, 12), (13, 16), (17, 20), (21, 24), (25, 28), (29, 32), (33, 36), (37, 40), (41, 50), (51, 60), (61, 80), (81, 400)]
acc = {s: {b: [0.0, 0.0, 0] for b in bands} for s in ('us', 'opp')}
for g in games:
    b = chess.Board(g['fen']); w = b.turn; fm0 = b.fullmove_number; cl = {'us': 10000.0, 'opp': 10000.0}
    for k, ms in enumerate(g['ms']):
        fm = fm0 + (k + (0 if w else 1)) // 2; side = 'us' if ((w if k % 2 == 0 else not w) == g['us_white']) else 'opp'
        for bd in bands:
            if bd[0] <= fm <= bd[1]: a = acc[side][bd]; a[0] += ms; a[1] += cl[side]; a[2] += 1
        cl[side] += 100 - ms
print('move numbers: ms a move, clock before the move (s), clock / (ms - 75)   -- ours | the opponent\'s')
for bd in bands:
    u, o = acc['us'][bd], acc['opp'][bd]
    if u[2]: print(f'  {bd[0]:3d}-{bd[1]:3d}: {u[0] / u[2]:5.0f} {u[1] / u[2] / 1000:5.2f} {u[1] / max(1, u[0] - 75 * u[2]):6.1f}  |  {o[0] / o[2]:5.0f} {o[1] / o[2] / 1000:5.2f} {o[1] / max(1, o[0] - 75 * o[2]):6.1f}')
