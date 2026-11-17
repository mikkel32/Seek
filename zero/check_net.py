"""The engine's evaluation against an independent re-computation from the trainer's weights, on random positions.

A new term in the evaluator has to be written twice (trainer, engine); this says whether the two agree. Plain, mirrored and
king-bucketed inputs (the bucket scheme is read off the size of the input table).

  python3 zero/check_net.py <engine binary> <network .bin> <the same network's .ema.pt> [positions=300] [mirror]
"""
import random, subprocess, sys
import chess, numpy as np, torch

binary, net_bin, net_pt = sys.argv[1:4]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 300; MIR = len(sys.argv) > 5 and sys.argv[5] == 'mirror'
w = {k: v.numpy().astype(np.float64) for k, v in torch.load(net_pt, map_location='cpu').items()}
H = w['b'].shape[0]; prod = 'qo' in w; nb = w['wo'].shape[0]; mlp = 'L1' in w; HP = []
PARTS = []                                   # how much of each evaluation the product terms supply (a check that agrees only because they are tiny proves nothing)
PT_W = np.repeat(np.array([0, 1, 1, 2, 4, 0] * 2, float), 64)
NKB = (w['E'].shape[0] - 1) // 768
KBT = [0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5] + [6] * 16
if NKB == 16:
    KBT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13] + [14, 14, 15, 15] * 3
if NKB == 32:
    KBT = list(range(32))
if 'Esh' in w:                               # shared table + per-bucket corrections: the engine's file holds their sum
    w['E'][:NKB * 768] += np.tile(w['Esh'][:768], (NKB, 1))
KB11 = [0, 0, 1, 2, 3, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8] + [9] * 16 + [10] * 32


def kbf(f):
    """a view's plain feature indices -> rows of the king-bucketed input table (the view's own king is piece code 5)"""
    if NKB == 1:
        return f
    ks = int(f[(f >= 320) & (f < 384)][0]) - 320
    if NKB == 11:
        return KB11[ks] * 768 + f
    mir = 7 if (ks & 7) >= 4 else 0; kq = ks ^ mir
    return KBT[(kq >> 3) * 4 + (kq & 7)] * 768 + (f // 64) * 64 + ((f % 64) ^ mir)


def feats(b):
    """feature indices in the mover's view and in the opponent's: own pieces 0-5, the other side's 6-11, the board turned so the viewer plays up"""
    own, opp = [], []
    for sq, p in b.piece_map().items():
        mine = p.color == b.turn; s = sq if b.turn else sq ^ 56; c = p.piece_type - 1
        own.append((c if mine else c + 6) * 64 + s); opp.append((c + 6 if mine else c) * 64 + (s ^ 56))
    return np.array(own), np.array(opp)


def ev(b):
    o, p = feats(b)
    hm = lambda f: f ^ 7 if MIR and (int(f[(f >= 320) & (f < 384)][0]) & 7) < 4 else f      # mirrored inputs: the view's own king on files a-d flips the files
    co = (w['b'] + w['E'][kbf(hm(o))].sum(0)).clip(0, 1); cp = (w['b'] + w['E'][kbf(hm(p))].sum(0)).clip(0, 1)
    bk = min(nb - 1, (len(o) - 2) // 4)
    if nb in (16, 32):                       # the queens join the key (own queen: piece code 4, the opponent's: 10, in the mover's view)
        pcb = min(7, (len(o) - 2) // 4); mq = int(any(256 <= f < 320 for f in o)); oq = int(any(640 <= f < 704 for f in o))
        bk = pcb * 2 + int(mq or oq) if nb == 16 else pcb * 4 + mq * 2 + oq
    lin = w['c'] + w['S'][o].sum(0); ph = min(1.0, PT_W[o].sum() / 24)
    out = ph * lin[0] + (1 - ph) * lin[1] + w['cb'][bk] + (co ** 2) @ w['wo'][bk] + (cp ** 2) @ w['wp'][bk]
    pr = 0.0
    if prod:
        h = H // 2; pr = (co[:h] * co[h:]) @ w['qo'][bk] + (cp[:h] * cp[h:]) @ w['qp'][bk] + (co * cp) @ w['qx'][bk]
    if mlp:                                  # the layered read-out of this bucket
        h1 = w['L1'].shape[0] // nb; hk = w['L1'].shape[1] // 2; x = np.concatenate([(co ** 2)[:hk], (cp ** 2)[:hk]])
        u = (w['L1'][bk * h1:(bk + 1) * h1] @ x + w['b1'][bk * h1:(bk + 1) * h1]).clip(0, 1) ** 2
        hd = w['L3'][bk] @ (w['L2'][bk] @ u + w['b2'][bk]).clip(0, 1); HP.append(abs(hd) * 271.58); pr = pr + hd
    PARTS.append(abs(pr) * 271.58)
    return (out + pr) * 271.58


random.seed(3); boards = []
while len(boards) < n:
    b = chess.Board()
    for _ in range(random.randint(4, 140)):
        if b.is_game_over():
            break
        b.push(random.choice(list(b.legal_moves)))
    if not b.is_game_over():
        boards.append(b)
text = ''.join(f'position fen {b.fen()}\neval\n' for b in boards) + 'quit\n'
got = [int(l.split()[1]) for l in subprocess.run([binary, net_bin], input=text, capture_output=True, text=True).stdout.split('\n') if l.startswith('eval ')]
want = np.array([ev(b) for b in boards]); got = np.array(got, float)
assert len(got) == n, f'the engine answered {len(got)} of {n}'
d = np.abs(got - np.clip(want, -20000, 20000))
print(f'{n} random positions, hidden {H}, king buckets {NKB}, product terms {"on" if prod else "off"}: largest difference {d.max():.2f} centipawns, mean {d.mean():.3f}; '
      f'evaluations range {got.min():.0f} to {got.max():.0f}, the product terms supply {np.mean(PARTS):.0f} centipawns on average'
      + (f', the layered read-out {np.mean(HP):.0f}' if mlp else ''))
sys.exit(0 if d.max() <= 1.5 else 1)
