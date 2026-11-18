import sys, importlib.util, torch, numpy as np
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
old = load('/root/loopchess/model.py', 'm_old'); new = load('/root/lmtest/model.py', 'm_new')
ck = torch.load('/root/loopchess/ckpt/B.pt', weights_only=False); dev = torch.device('cuda')
a = old.Ouro(**ck['cfg']).to(dev); a.load_state_dict(ck['ema']); a.eval(); b = new.Ouro(**ck['cfg']).to(dev); b.load_state_dict(ck['ema']); b.eval()
bd = torch.from_numpy(np.ascontiguousarray(np.load('/root/loopchess/data/lm_0_b.npy', mmap_mode='r')[:256])).to(dev).long(); me = np.load('/root/loopchess/data/lm_0_m.npy', mmap_mode='r')[:256]
ca = torch.from_numpy(np.ascontiguousarray(me[:, 0])).to(dev).long(); ep = torch.from_numpy(np.ascontiguousarray(me[:, 1])).to(dev).long()
with torch.no_grad():
    pa, va, qa, ma = a(bd, ca, ep, loops=2, tbptl=False); pb, vb, qb, mb = b(bd, ca, ep, loops=2, tbptl=False)
la = pa.masked_fill(~ma, -1e9).log_softmax(1); lb = pb.masked_fill(~mb, -1e9).log_softmax(1)
print('same best move:', float((la.argmax(1) == lb.argmax(1)).float().mean()), '| largest policy log-prob difference on legal moves:', float(((la - lb).abs() * ma).max()),
      '| value logit difference: mean', float((va - vb).abs().mean()), 'max', float((va - vb).abs().max()), '| masks equal:', bool((ma == mb).all()))
