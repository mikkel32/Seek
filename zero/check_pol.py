"""The engine's "where to search" scores against a re-computation from the trainer's weights, on random positions.

  python3 zero/check_pol.py <engine binary> <network .bin with the POL1 trailer> <evaluator .ema.pt> <read-out .pol.pt> [positions=200]
"""
import random, subprocess, sys
import chess, numpy as np, torch
binary, net_bin, net_pt, pol_pt = sys.argv[1:5]; n = int(sys.argv[5]) if len(sys.argv) > 5 else 200
w = {k: v.numpy().astype(np.float64) for k, v in torch.load(net_pt, map_location='cpu').items()}; p = {k: v.numpy().astype(np.float64) for k, v in torch.load(pol_pt, map_location='cpu').items()}
random.seed(4); boards = []
while len(boards) < n:
    b = chess.Board()
    for _ in range(random.randint(6, 100)):
        if b.is_game_over():
            break
        b.push(random.choice(list(b.legal_moves)))
    if not b.is_game_over() and not b.is_check():
        boards.append(b)


def scores(b):
    o, q = [], []
    for sq, pc in b.piece_map().items():
        mine = pc.color == b.turn; s = sq if b.turn else sq ^ 56; c = pc.piece_type - 1
        o.append((c if mine else c + 6) * 64 + s); q.append((c + 6 if mine else c) * 64 + (s ^ 56))
    h = np.concatenate([(w['b'] + w['E'][o].sum(0)).clip(0, 1) ** 2, (w['b'] + w['E'][q].sum(0)).clip(0, 1) ** 2])
    out = p['l2.weight'] @ np.maximum(p['l1.weight'] @ h + p['l1.bias'], 0) + p['l2.bias']; res = {}
    for m in b.legal_moves:
        if b.is_capture(m) or m.promotion:
            continue
        f = m.from_square if b.turn else m.from_square ^ 56; t = m.to_square if b.turn else m.to_square ^ 56
        res[m.uci()] = 256 * (out[f] + out[64 + (b.piece_type_at(m.from_square) - 1) * 64 + t])
    return res


text = ''.join(f'position fen {b.fen()}\npol\n' for b in boards) + 'quit\n'
out = subprocess.run([binary, net_bin], input=text, capture_output=True, text=True).stdout.split('\n'); i = 0; diffs = []; first = 0; cnt = 0
for b in boards:
    got = {}
    while i < len(out) and not out[i].startswith('pol end'):
        if out[i].startswith('pol '):
            _, mv, sc = out[i].split(); got[mv] = int(sc)
        i += 1
    i += 1; want = scores(b)
    for mv, sc in got.items():
        if mv in want:
            diffs.append(abs(sc - want[mv]))
    if got and want:
        cnt += 1; first += max(got, key=got.get) == max(want, key=want.get)
    # (the engine also lists moves that leave its king in check: its generator is pseudo-legal, the search rejects them later)
d = np.array(diffs)
print(f'{cnt} positions, {len(d)} quiet moves: engine against re-computation, mean difference {d.mean() / 256:.4f} (in units where the scores spread over {np.std([v for b in boards[:50] for v in scores(b).values()]) / 256:.2f}), largest {d.max() / 256:.3f}; same favourite move in {first / cnt:.1%} of positions')
