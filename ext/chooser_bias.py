import sys, io, contextlib, importlib.util, numpy as np
spec = importlib.util.spec_from_file_location('we', 'ext/whose_eval.py'); we = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(we)
y, res, ok = we.y, we.res, we.ok
lab = 271.58 * np.log(np.clip(y, 1e-4, 1 - 1e-4) / (1 - np.clip(y, 1e-4, 1 - 1e-4)))        # the label back in centipawns
print(f'{ok.sum()} positions from our own engine\'s games; the side to move is the one that did NOT choose the position')
for name, e in res.items():
    m = ok & (np.abs(lab) < 300)
    # least-squares scale of this evaluator against the label, then the signed error for the mover
    k = float((e[m] * lab[m]).sum() / (e[m] ** 2).sum()); d = k * e[m] - lab[m]
    print(f'{name}: scaled by {k:.2f}; mean signed error for the side to move {d.mean():+.1f} cp (negative = the evaluator likes the position more for the side that chose it than the deep search does), '
          f'spread {d.std():.1f} cp, share of errors beyond 100 cp {100 * (np.abs(d) > 100).mean():.1f}%, beyond 200 cp {100 * (np.abs(d) > 200).mean():.2f}%, correlation {np.corrcoef(e[m], lab[m])[0, 1]:.3f}')
