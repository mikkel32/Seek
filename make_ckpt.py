"""Freeze the EMA weights of a training checkpoint into a clean engine checkpoint."""
import sys, torch
ck = torch.load(sys.argv[1], weights_only=False)
out = dict(cfg=ck['cfg'], ema={k: v.clone() for k, v in ck['ema'].items()}, beta=4.0, step=ck['step'], seen=ck['seen'])
torch.save(out, sys.argv[2])
print(sys.argv[2], 'step', ck['step'], 'seen', ck['seen'], 'params', sum(v.numel() for v in out['ema'].values()))
