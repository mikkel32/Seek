"""What is one unit of the stored score worth? The same positions through our own evaluator (which speaks the training labels' scale: 271.58 units per logit)."""
import subprocess, sys, numpy as np
raw = sys.stdin.buffer.read(400_000 * 32); n = len(raw) // 32; r = np.frombuffer(raw, np.uint8, n * 32).reshape(n, 32)
score = r[:, 24:26].copy().view('<i2').reshape(-1).astype(np.float64); res = r[:, 26].astype(np.float64) / 2
PC = 'PNBRQK??pnbrqk??'
def fen(rec):
    occ = int.from_bytes(bytes(rec[:8]), 'little'); board = [''] * 64; k = 0
    for s in range(64):
        if occ >> s & 1:
            board[s] = PC[(rec[8 + k // 2] >> (4 * (k & 1))) & 15]; k += 1
    rows = []
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
fens = [fen(x) for x in r]; print(fens[0], score[0], res[0]); print(fens[1], score[1], res[1])
text = ''.join(f'position fen {f}\neval\n' for f in fens) + 'quit\n'
out = subprocess.run(sys.argv[1:], input=text, capture_output=True, text=True).stdout
e = np.array([int(l.split()[1]) for l in out.split('\n') if l.startswith('eval ')], float); assert len(e) == n, (len(e), n)
print('rank correlation of the stored score with our evaluation: %.3f' % np.corrcoef(np.argsort(np.argsort(score)), np.argsort(np.argsort(e)))[0, 1])
print('stored score band -> median of our evaluation (units of 271.58 per logit), ratio, mean game result')
for lo, hi in ((5, 20), (20, 40), (40, 70), (70, 110), (110, 160), (160, 230), (230, 320), (320, 450), (450, 700), (700, 1200), (1200, 3000), (3000, 40000)):
    m = (np.abs(score) >= lo) & (np.abs(score) < hi); s = np.sign(score[m]); em = np.median(e[m] * s); sm = np.median(np.abs(score[m]))
    print('  %5d..%5d: %5.1f%%  our median %7.1f  ratio %.2f  result %.3f  our evaluation as a win chance %.3f' % (lo, hi, 100 * m.mean(), em, em / sm, (0.5 + (res[m] - 0.5) * s).mean(), 1 / (1 + np.exp(-em / 271.58))))
