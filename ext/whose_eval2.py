"""Follow-up to whose_eval.py: is the endgame difference resolution or only scale? Squared error by piece count, with the
best scale fitted separately for every evaluator AND piece-count group, on all labels, and split by the label.

  python3 ext/whose_eval2.py <positions> <tag> "<name>|<stockfish>|<options>" ... "<name>|<our engine> <network>" ...
"""
import sys
sys.argv, rest = sys.argv[:3] + [], sys.argv[3:]
import importlib.util, io, contextlib
spec = importlib.util.spec_from_file_location('we', 'ext/whose_eval.py'); we = importlib.util.module_from_spec(spec)
sys.argv = sys.argv[:3] + rest
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(we)
import numpy as np
y, res, ok, pieces = we.y, we.res, we.ok, we.pieces
scales = np.geomspace(40, 3000, 240)
for lo, hi in ((2, 6), (7, 12), (13, 22), (23, 32)):
    m = ok & (pieces >= lo) & (pieces <= hi); t = y[m]
    print(f'{lo}-{hi} pieces: {m.sum()} positions ({100 * m.sum() / ok.sum():.0f}% of all), {100 * (np.abs(t - 0.5) < 0.2).mean():.0f}% of them balanced, {100 * (t == 0.5).mean():.0f}% exactly level')
    for name, e in res.items():
        def mse(s, mm=m):
            p = 1 / (1 + np.exp(-np.clip(e[mm] / s, -40, 40))); return float(((p - y[mm]) ** 2).mean())
        s = min(scales, key=mse); p = 1 / (1 + np.exp(-np.clip(e[m] / s, -40, 40))); b = np.abs(t - 0.5) < 0.2
        own = np.abs(p - 0.5) < 0.2                         # the evaluator itself calls it balanced
        print(f'    {name}: scale {s:.0f}; squared error {mse(s):.5f}; mean error {np.abs(p - t).mean():.4f} (label balanced {np.abs(p - t)[b].mean():.4f}, label decided {np.abs(p - t)[~b].mean():.4f}); '
              f'where it says balanced itself ({100 * own.mean():.0f}%): {np.abs(p - t)[own].mean():.4f}; where it says 70-90% for one side: label spread {np.std(t[(np.abs(p - 0.5) >= 0.2) & (np.abs(p - 0.5) < 0.4)]):.3f}')
