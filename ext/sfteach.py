"""Stockfish 14.1's own network as a teacher for training: its evaluation of a batch of positions, on the graphics card.

The network is the one embedded in the installed Stockfish binary (ext/sfnet_extract.py); this is our re-implementation of
its arithmetic in matrix form (the same arithmetic as ext/sfnet_ref.py and the engine's -DSFNET build, which equal the
engine's `eval` output). Positions come in the trainer's form: 32 feature numbers in the mover's view (piece code x 64 +
square, own pieces 0-5, the other side's 6-11, 768 = empty). Labels made this way are Stockfish's judgments.

  load(path, device) -> teacher;  teacher(idx) -> (raw network value, value as Stockfish's search sees it), both in
  Stockfish's internal units from the mover's point of view.
  python3 ext/sfteach.py <network file> <engine with the sfraw command> [positions]   checks against the -DSFNET engine
"""
import struct, sys
import numpy as np, torch
import torch.nn.functional as F


def load(path, dev):
    b = open(path, 'rb').read(); o = 12 + struct.unpack('<I', b[8:12])[0] + 4; H, NF = 1024, 22528
    bias = torch.from_numpy(np.frombuffer(b, '<i2', H, o).astype(np.float32)); o += 2 * H
    W = torch.from_numpy(np.frombuffer(b, '<i2', NF * H, o).reshape(NF, H).astype(np.float32)); o += 2 * NF * H
    PS = torch.from_numpy(np.frombuffer(b, '<i4', NF * 8, o).reshape(NF, 8).astype(np.float32)); o += 4 * NF * 8
    b0, w0, b1, w1, b2, w2 = [], [], [], [], [], []
    for k in range(8):
        o += 4
        b0.append(np.frombuffer(b, '<i4', 8, o)); o += 32; w0.append(np.frombuffer(b, 'i1', 8 * 2048, o).reshape(8, 2048)); o += 8 * 2048
        b1.append(np.frombuffer(b, '<i4', 32, o)); o += 128; w1.append(np.frombuffer(b, 'i1', 32 * 32, o).reshape(32, 32)[:, :8]); o += 32 * 32
        b2.append(np.frombuffer(b, '<i4', 1, o)); o += 4; w2.append(np.frombuffer(b, 'i1', 32, o)); o += 32
    T = lambda x: torch.from_numpy(np.array(x).astype(np.float32)).to(dev)
    W = torch.cat([W, torch.zeros(1, H)]).to(dev); PS = torch.cat([PS, torch.zeros(1, 8)]).to(dev); bias = bias.to(dev)
    B0, W0, B1, W1, B2, W2 = T(b0), T(w0), T(b1), T(w1), T(b2)[:, 0], T(w2)          # (8,8) (8,8,2048) (8,32) (8,32,8) (8,) (8,32)
    MG = torch.tensor([0, 781, 825, 1276, 2538, 0, 0, 781, 825, 1276, 2538, 0, 0], dtype=torch.float32, device=dev)   # by piece code; 12 = empty

    @torch.no_grad()
    def teacher(idx):
        idx = idx.long(); Bn = idx.shape[0]; empty = idx >= 768; code = torch.where(empty, torch.full_like(idx, 12), idx // 64); sq = idx % 64
        acc, psq = [], []
        for view in (0, 1):                                   # 0 = the mover (White in this view), 1 = the other side
            kcode = 5 if view == 0 else 11
            ksq = (sq * (code == kcode)).sum(1)                # that side's king square
            flip = 0 if view == 0 else 56; ko = ksq ^ flip; mir = ((ko & 7) < 4).long() * 7; ko = ko ^ mir
            kb = (7 - (ko >> 3)) * 4 + (7 - (ko & 7))
            mine = (code < 6) if view == 0 else ((code >= 6) & (code < 12)); kind = code % 6
            plane = torch.where(kind == 5, torch.full_like(code, 10), 2 * kind + (~mine).long())
            f = (sq ^ flip ^ mir[:, None]) + plane * 64 + 704 * kb[:, None]
            f = torch.where(empty, torch.full_like(f, 22528), f)
            acc.append(bias + F.embedding_bag(f, W, mode='sum', padding_idx=22528)); psq.append(F.embedding_bag(f, PS, mode='sum', padding_idx=22528))
        n = (~empty).sum(1); bk = ((n - 1) // 4).clamp(0, 7)
        x = torch.cat([acc[0].clamp(0, 127), acc[1].clamp(0, 127)], 1)
        ar = torch.arange(Bn, device=idx.device)
        h0 = torch.floor(((x @ W0.reshape(64, 2048).T).view(Bn, 8, 8)[ar, bk] + B0[bk]) / 64).clamp(0, 127)
        h1 = torch.floor((torch.bmm(W1[bk], h0[:, :, None]).squeeze(2) + B1[bk]) / 64).clamp(0, 127)
        out = (W2[bk] * h1).sum(1) + B2[bk]
        ps = torch.trunc((psq[0][ar, bk] - psq[1][ar, bk]) / 2)
        raw = torch.trunc((ps + out) / 16)
        npm_me = (MG[code] * (code < 6)).sum(1); npm_you = (MG[code] * ((code >= 6) & (code < 12))).sum(1)
        adj = torch.where((npm_me - npm_you).abs() <= 1276 - 825, torch.trunc(torch.trunc((121 * ps + 135 * out) / 128) / 16), raw)
        pawns = ((code == 0) | (code == 6)).sum(1)
        scale = 883 + 32 * pawns + torch.floor(32 * (npm_me + npm_you) / 1024)
        return raw, torch.trunc(adj * scale / 1024)
    return teacher


if __name__ == '__main__':
    import os, subprocess
    path, engine = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    X = np.load('data/ev_ival_x.npy', mmap_mode='r')[:N].astype(np.int64); PCN = 'PNBRQKpnbrqk'; fens = []
    for r in X:
        board = [''] * 64; rows = []
        for i in r:
            if i < 768: board[i % 64] = PCN[i // 64]
        for rank in range(7, -1, -1):
            row = ''; gap = 0
            for f in range(8):
                c = board[rank * 8 + f]
                if c: row += (str(gap) if gap else '') + c; gap = 0
                else: gap += 1
            rows.append(row + (str(gap) if gap else ''))
        fens.append('/'.join(rows) + ' w - - 0 1')
    out = subprocess.run(engine.split(), input=''.join(f'position fen {f}\nsfraw\n' for f in fens) + 'quit\n', capture_output=True, text=True, env=dict(os.environ, SFNET_FILE=path)).stdout
    eng = np.array([int(l.split()[1]) for l in out.split('\n') if l.startswith('sfraw ')], float)
    t = load(path, torch.device('cpu')); raw, fin = t(torch.from_numpy(X)); raw = raw.numpy()
    print(f'{N} positions: teacher function against the -DSFNET engine: largest difference {np.abs(raw - eng).max():.1f} internal units, exact in {100 * (raw == eng).mean():.1f}%; final value ranges {fin.min():.0f} to {fin.max():.0f}')
