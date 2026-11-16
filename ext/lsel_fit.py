"""Does a learned model tell which late moves matter better than the standing reduction rule?  (records of the -DLSEL build)

  python3 ext/lsel_fit.py <records.bin> [max records] [--export=<model file for the engine's SelFile option>]

Three predictors of "a full-depth look at this move beats the bound", fitted on four fifths of the searches and judged on
the rest:  standing = what the engine's own rule knows (its reduction, the depth, which pruning rule would have fired);
additive = one weight per bucket of every recorded feature;  network = a small two-layer network on the same buckets.
Printed: log-loss, area under the curve, and how the moves can be split at equal risk: the share of all late moves that
can be put in a low-risk group that holds no more of the moves that matter than the standing rule's own most-reduced group.
"""
import sys
import numpy as np, torch

DT = np.dtype([('hs', '<i2'), ('hmain', '<i2'), ('hc1', '<i2'), ('hc2', '<i2'), ('ev_alpha', '<i2'), ('v_alpha', '<i2'), ('caph', '<i2'), ('see', '<i2'),
               ('tt_alpha', '<i2'), ('best_alpha', '<i2'), ('depth', 'u1'), ('legal', 'u1'), ('ply', 'u1'), ('rstd', 'u1'), ('nd', 'u1'), ('rootd', 'u1'),
               ('pmcount', 'u1'), ('nq', 'u1'), ('nc', 'u1'), ('piece', 'u1'), ('victim', 'u1'), ('prune', 'u1'), ('ttdepth', 'u1'), ('ttflag', 'u1'),
               ('from', 'u1'), ('to', 'u1'), ('f1', 'u1'), ('f2', 'u1'), ('gid', '<u2')])
args = [a for a in sys.argv[1:] if not a.startswith('--')]; EXPORT = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--export=')), '')
r = np.fromfile(args[0], dtype=DT)
if len(args) > 1 and len(r) > int(args[1]):
    r = r[np.random.RandomState(0).choice(len(r), int(args[1]), replace=False)]
y = ((r['f2'] >> 7) & 1).astype(np.float32); val = (r['gid'] % 5) == 0
print(f'{len(r)} records, {int(y.sum())} beat the bound ({100 * y.mean():.2f}%); judged on {int(val.sum())}')


def qb(x, n, mask=None):
    """quantile buckets of a numeric feature (edges from the training part); mask = rows that get a bucket of their own"""
    ok = ~val if mask is None else (~val & ~mask)
    e = np.unique(np.quantile(x[ok], np.linspace(0, 1, n + 1)[1:-1])); b = np.searchsorted(e, x, side='right')
    if mask is not None:
        b = np.where(mask, len(e) + 1, b)
    return b.astype(np.int64)


# every feature in the order the engine reads the model file: (name, record field, kind, n)
#   cap = the value itself up to n; q = n quantile buckets; qs = the same with a bucket of its own for "none" (-32000); bit = bit n of the field
SPEC = [('depth', 'depth', 'cap', 20), ('move number', 'legal', 'q', 16), ('ply', 'ply', 'q', 8), ('history', 'hs', 'q', 16), ('main history', 'hmain', 'q', 12),
        ('reply history', 'hc1', 'q', 12), ('follow-up history', 'hc2', 'q', 12), ('evaluation minus bound', 'ev_alpha', 'qs', 16), ('exchange value', 'see', 'q', 8),
        ('capture history', 'caph', 'q', 8), ('table score minus bound', 'tt_alpha', 'qs', 8), ('best so far minus bound', 'best_alpha', 'qs', 8), ('quiet moves tried', 'nq', 'q', 8),
        ('captures tried', 'nc', 'cap', 7), ("parent's move number", 'pmcount', 'q', 8), ('piece', 'piece', 'cap', 5), ('victim', 'victim', 'cap', 6), ('table bound kind', 'ttflag', 'cap', 3),
        ('depth left after the move', 'nd', 'cap', 20),
        ('pv node', 'f1', 'bit', 0), ('cut node', 'f1', 'bit', 1), ('improving', 'f1', 'bit', 2), ('in check', 'f1', 'bit', 3), ('gives check', 'f1', 'bit', 4), ('quiet', 'f1', 'bit', 5),
        ('killer', 'f1', 'bit', 6), ('table move exists', 'f1', 'bit', 7), ('table move is a capture', 'f2', 'bit', 0), ('losing capture', 'f2', 'bit', 1), ('promotion', 'f2', 'bit', 2),
        ('leaves an attacked square', 'f2', 'bit', 3), ('goes to an attacked square', 'f2', 'bit', 4), ('goes where a pawn attacks', 'f2', 'bit', 5), ('table says below bound', 'f2', 'bit', 6)]
F = {}; EDGES = {}
for name, field, kind, n in SPEC:
    x = r[field].astype(np.int32)
    if kind == 'cap':
        F[name] = np.minimum(x, n).astype(np.int64)
    elif kind == 'bit':
        F[name] = ((x >> n) & 1).astype(np.int64)
    else:
        none = (x == -32000) if kind == 'qs' else None; ok = ~val if none is None else (~val & ~none)
        e = np.unique(np.quantile(x[ok], np.linspace(0, 1, n + 1)[1:-1]).round().astype(np.int64)); b = np.searchsorted(e, x, side='right')
        F[name] = (np.where(none, len(e) + 1, b) if none is not None else b).astype(np.int64); EDGES[name] = e
STAND = {'standing reduction': np.minimum(r['rstd'], 8).astype(np.int64), 'depth': F['depth'], 'pruning rule': r['prune'].astype(np.int64)}


def onehot_index(feats):
    off = 0; cols = []
    for k, v in feats.items():
        cols.append(v + off); off += int(v.max()) + 1
    return np.stack(cols, 1), off


def fit(feats, hidden=0, epochs=6, name=''):
    X, n = onehot_index(feats); Xt = torch.from_numpy(X); yt = torch.from_numpy(y); tr = torch.from_numpy(np.nonzero(~val)[0]); va = torch.from_numpy(np.nonzero(val)[0])
    torch.manual_seed(0)
    emb = torch.nn.EmbeddingBag(n, hidden or 1, mode='sum'); head = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(hidden, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)) if hidden else None
    bias = torch.nn.Parameter(torch.tensor([float(np.log(y.mean() / (1 - y.mean())))]))
    torch.nn.init.zeros_(emb.weight) if not hidden else None
    ps = list(emb.parameters()) + [bias] + (list(head.parameters()) if head else []); opt = torch.optim.Adam(ps, lr=0.01 if not hidden else 0.003)
    fwd = lambda idx: (head(emb(Xt[idx])).squeeze(1) if hidden else emb(Xt[idx]).squeeze(1)) + bias
    for ep in range(epochs):
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), 8192):
            b = perm[i:i + 8192]; loss = torch.nn.functional.binary_cross_entropy_with_logits(fwd(b), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
        for g in opt.param_groups:
            g['lr'] *= 0.6
    with torch.no_grad():
        z = torch.cat([fwd(va[i:i + 65536]) for i in range(0, len(va), 65536)]).numpy()
    return z, (emb, bias, feats)


def auc(z, t):
    o = np.argsort(z); rk = np.empty(len(z)); rk[o] = np.arange(1, len(z) + 1); p = t.sum(); n = len(t) - p
    return (rk[t == 1].sum() - p * (p + 1) / 2) / (p * n)


yv = y[val]; ll = lambda z: float(np.mean(np.logaddexp(0, z) - yv * z))
base = float(-(yv.mean() * np.log(yv.mean()) + (1 - yv.mean()) * np.log(1 - yv.mean())))
res = {}
for name, feats, hid in (('standing', STAND, 0), ('additive', F, 0), ('network', F, 48)):
    z, model = fit(feats, hid); res[name] = (z, model)
    print(f'{name:9s}: log-loss {ll(z):.5f} (knowing nothing: {base:.5f}; removed {100 * (1 - ll(z) / base):.1f}%), area under the curve {auc(z, yv):.4f}')

# the split at equal risk, by depth group
rv = r[val]
for lo, hi in ((2, 4), (5, 7), (8, 30)):
    m = (rv['depth'] >= lo) & (rv['depth'] <= hi); t = yv[m]; pos = t.sum()
    if pos < 20:
        continue
    print(f'depth {lo}-{hi}: {int(m.sum())} moves, {int(pos)} beat the bound ({100 * t.mean():.2f}%)')
    for share in (0.5, 0.7, 0.85):
        line = f'   the {int(100 * share)}% of moves judged least likely to matter hold'
        for name in ('standing', 'additive', 'network'):
            z = res[name][0][m]; k = int(share * len(z)); o = np.argsort(z, kind='stable')
            line += f'  {name} {100 * t[o[:k]].sum() / pos:.1f}%'
        print(line + ' of those that do')
# what the additive model leans on
emb, bias, feats = res['additive'][1]; w = emb.weight.detach().numpy()[:, 0]; off = 0; imp = []
Xv, _ = onehot_index(feats)
for k, v in feats.items():
    n = int(v.max()) + 1; imp.append((float(np.std(w[v[val] + off])), k)); off += n
print('additive model, spread of each feature\'s contribution (logit units): ' + ', '.join(f'{k} {s:.2f}' for s, k in sorted(imp, reverse=True)[:14]))

if EXPORT:                                                   # the additive model as the engine reads it: bias, then per feature its edges and its weights
    off = 0; lines = [f'{float(bias.detach()[0]):.6f}']
    for name, field, kind, n in SPEC:
        nb = int(F[name].max()) + 1; e = EDGES.get(name, [])
        lines.append(f'{len(e)} ' + ' '.join(str(int(x)) for x in e) + f' | {nb} ' + ' '.join(f'{x:.5f}' for x in w[off:off + nb])); off += nb
    open(EXPORT, 'w').write('\n'.join(lines) + '\n'); print(f'additive model -> {EXPORT}')
    z = res['additive'][0]; rs = rv['rstd']
    for lo, hi in ((3, 4), (5, 7), (8, 11), (12, 30)):
        m = (rv['depth'] >= lo) & (rv['depth'] <= hi) & (rv['legal'] > 1)
        print(f'depth {lo}-{hi}: logit quantiles 10/50/90% {np.quantile(z[m], 0.1):.2f} {np.quantile(z[m], 0.5):.2f} {np.quantile(z[m], 0.9):.2f}; standing reduction mean {rs[m].mean():.2f}, '
              f'by logit third: {rs[m][z[m] < np.quantile(z[m], 1 / 3)].mean():.2f} / {rs[m][(z[m] >= np.quantile(z[m], 1 / 3)) & (z[m] < np.quantile(z[m], 2 / 3))].mean():.2f} / {rs[m][z[m] >= np.quantile(z[m], 2 / 3)].mean():.2f}')
