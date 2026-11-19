"""Stockfish training records (bullet format, 32 bytes) -> our 24-byte form; checks on a sample:
our own network's evaluation against the stored score (wrong orientation or piece codes would destroy the correlation),
and which scale turns the stored score into a winning chance that fits the game results best."""
import sys, numpy as np, torch
sys.path.insert(0, 'zero'); from compact import decode_torch
LUTN = np.array([0, 1, 2, 3, 4, 5, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0], np.uint8)


def to_compact(r):
    out = np.empty((len(r), 24), np.uint8); out[:, :8] = r[:, :8]; p = r[:, 8:24]; out[:, 8:] = LUTN[p & 15] | (LUTN[p >> 4] << 4)
    return out


if __name__ == '__main__':
    raw = np.fromfile(sys.argv[1], dtype=np.uint8); n = min(len(raw) // 32, 1_000_000); r = raw[:n * 32].reshape(n, 32)
    score = r[:, 24:26].copy().view('<i2').reshape(-1).astype(np.float64); res = r[:, 26].astype(np.float64) / 2
    DEV = 'cuda'; w = {k: v.to(DEV).float() for k, v in torch.load(sys.argv[2], map_location='cpu').items()}
    _i = np.arange(768); LUT = torch.from_numpy(np.concatenate([((_i // 64 + 6) % 12) * 64 + ((_i % 64) ^ 56), [768]]).astype(np.int64)).to(DEV)
    PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64), torch.zeros(1)]).to(DEV)
    mh = lambda idx: torch.zeros(idx.shape[0], 769, device=DEV).scatter_add_(1, idx, torch.ones(idx.shape, device=DEV))[:, :768]
    outs = []
    with torch.no_grad():
        C = torch.from_numpy(to_compact(r)).to(DEV)
        for j in range(0, n, 16384):
            idx = decode_torch(C[j:j + 16384]).long(); Mo, Mp = mh(idx), mh(LUT[idx]); nb = w['wo'].shape[0]; bk = ((Mo.sum(1) - 2) // 4).clamp(0, nb - 1).long()
            xo = (w['b'] + Mo @ w['E'][:768]).clamp(0, 1) ** 2; xp = (w['b'] + Mp @ w['E'][:768]).clamp(0, 1) ** 2
            lin = w['c'] + Mo @ w['S'][:768]; ph = ((Mo @ PT_W[:768]) / 24).clamp(max=1.0)
            outs.append((ph * lin[:, 0] + (1 - ph) * lin[:, 1] + w['cb'][bk] + (xo * w['wo'][bk]).sum(1) + (xp * w['wp'][bk]).sum(1)).cpu().numpy())
    o = np.concatenate(outs) * 271.58; m = np.abs(score) < 1500
    print(f'{n} records; our evaluation against the stored score (|score| < 1500): correlation {np.corrcoef(o[m], score[m])[0, 1]:.3f}; our evaluation against the game result: {np.corrcoef(o, res)[0, 1]:.3f}; stored score against the result: {np.corrcoef(np.clip(score, -2000, 2000), res)[0, 1]:.3f}')
    sl = np.polyfit(score[m], o[m], 1)[0]; print(f'one unit of the stored score = {sl:.2f} of our centipawns')
    best = None
    for K in (200, 240, 280, 320, 360, 400, 450, 500, 600):
        p = 1 / (1 + np.exp(-score / K)); ce = -(res * np.log(p + 1e-9) + (1 - res) * np.log(1 - p + 1e-9)).mean(); best = min(best or (9, 0), (ce, K)); print(f'  scale {K}: fit to results {ce:.4f}')
    print('best scale:', best[1])
