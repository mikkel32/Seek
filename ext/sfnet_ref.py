"""Reference re-computation of the installed Stockfish 14.1's network evaluation from the network embedded in its binary,
checked against the engine's own `eval` output. Layout found from the bytes: 22,528 inputs (32 mirrored king buckets x 704)
-> 1024 per side, 8 material buckets each with a material term and layers 2048 -> 8 -> 32 -> 1.
  python3 sfref.py <variant of the king-bucket numbering: 0 or 1> [positions=300]"""
import re, struct, subprocess, sys
import numpy as np
VAR = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
b = open('/root/ext/sf141_embedded.bin', 'rb').read(); o = 12 + struct.unpack('<I', b[8:12])[0] + 4
H, NF = 1024, 22528
bias = np.frombuffer(b, '<i2', H, o).astype(np.int64); o += 2 * H
W = np.frombuffer(b, '<i2', NF * H, o).reshape(NF, H); o += 2 * NF * H
PS = np.frombuffer(b, '<i4', NF * 8, o).reshape(NF, 8); o += 4 * NF * 8
stacks = []
for k in range(8):
    o += 4
    b0 = np.frombuffer(b, '<i4', 8, o).astype(np.int64); o += 32; w0 = np.frombuffer(b, 'i1', 8 * 2048, o).reshape(8, 2048).astype(np.int64); o += 8 * 2048
    b1 = np.frombuffer(b, '<i4', 32, o).astype(np.int64); o += 128; w1 = np.frombuffer(b, 'i1', 32 * 32, o).reshape(32, 32).astype(np.int64)[:, :8]; o += 32 * 32
    b2 = np.frombuffer(b, '<i4', 1, o).astype(np.int64); o += 4; w2 = np.frombuffer(b, 'i1', 32, o).astype(np.int64); o += 32
    stacks.append((b0, w0, b1, w1, b2, w2))
print('network read, bytes used', o)
PLANE = {'P': 0, 'N': 2, 'B': 4, 'R': 6, 'Q': 8, 'K': 10}       # x 64; +64 for the other side's piece; both kings share plane 10


def parse(fen):
    rows, stm = fen.split()[0].split('/'), fen.split()[1]; pcs = []
    for r, row in enumerate(rows):
        f = 0
        for c in row:
            if c.isdigit():
                f += int(c)
            else:
                pcs.append(((7 - r) * 8 + f, c)); f += 1
    return pcs, stm == 'w'


def evaluate(fen):
    pcs, wtm = parse(fen); acc = []; psq = []
    for persp_white in (True, False):
        ksq = [s for s, c in pcs if c == ('K' if persp_white else 'k')][0]
        flip = 0 if persp_white else 56
        mir = 7 if ((ksq ^ flip) & 7) < 4 else 0
        ko = ksq ^ flip ^ mir                                    # the viewer's king, on files e-h
        r_, f_ = ko >> 3, ko & 7
        kb = ((7 - r_) * 4 + (7 - f_)) if VAR == 0 else (r_ * 4 + (f_ - 4))
        idx = []
        for s, c in pcs:
            own = c.isupper() == persp_white; pl = PLANE[c.upper()] * 64 + (0 if (own or c.upper() == 'K') else 64)
            idx.append((s ^ flip ^ mir) + pl + 704 * kb)
        acc.append(bias + W[idx].astype(np.int64).sum(0)); psq.append(PS[idx].astype(np.int64).sum(0))
    us, them = (0, 1) if wtm else (1, 0); bk = (len(pcs) - 1) // 4
    x = np.concatenate([np.clip(acc[us], 0, 127), np.clip(acc[them], 0, 127)])
    b0, w0, b1, w1, b2, w2 = stacks[bk]
    h0 = np.clip((b0 + w0 @ x) >> 6, 0, 127); h1 = np.clip((b1 + w1 @ h0) >> 6, 0, 127); out = int(b2[0] + w2 @ h1)
    tz = lambda p, q: int(p / q) if True else 0                # C++ integer division truncates toward zero
    v = tz(tz(int(psq[us][bk] - psq[them][bk]), 2) + out, 16)
    return v if wtm else -v                                      # white's point of view, internal units


PCN = 'PNBRQKpnbrqk'


def mkfen(r):
    board = [''] * 64; rows = []
    for i in r:
        if i < 768:
            board[i % 64] = PCN[i // 64]
    for rank in range(7, -1, -1):
        row = ''; gap = 0
        for f in range(8):
            c = board[rank * 8 + f]
            if c:
                row += (str(gap) if gap else '') + c; gap = 0
            else:
                gap += 1
        rows.append(row + (str(gap) if gap else ''))
    return '/'.join(rows) + ' w - - 0 1'


X = np.load('/root/loopchess/data/ev_ival_x.npy', mmap_mode='r')[:N]
fens = []
for i, r in enumerate(X):
    f = mkfen(r)
    if i % 2:                                                    # every other position with Black to move: colours swapped, board turned
        rows = f.split()[0].split('/')[::-1]; f = '/'.join(row.swapcase() for row in rows) + ' b - - 0 1'
    fens.append(f)
p = subprocess.Popen(['/usr/games/stockfish'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
out, _ = p.communicate(''.join(f'position fen {f}\neval\n' for f in fens) + 'quit\n')
sf = [float(m) if m != 'none' else None for m in re.findall(r'NNUE evaluation:?\s+([-+]?[0-9.]+|none)', out)]
assert len(sf) == len(fens), (len(sf), len(fens))
d = [(evaluate(f) * 100 / 208, s * 100) for f, s in zip(fens, sf) if s is not None]
err = np.array([abs(a - s) for a, s in d]); print(f'variant {VAR}: {len(d)} positions, mean difference {err.mean():.3f} pawns/100, largest {err.max():.2f}; exact to the printed precision in {100 * (err < 0.6).mean():.1f}%')
print('samples (ours, engine):', [(round(a, 1), round(s)) for a, s in d[:6]], 'worst:', sorted([(round(abs(a - s), 1), round(a, 1), round(s)) for a, s in d])[-3:])
