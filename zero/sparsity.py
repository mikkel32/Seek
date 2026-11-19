"""How many hidden units are exactly zero (or one) after clipping, on real positions? (decides how the engine should read them)"""
import sys, numpy as np, torch
sys.path.insert(0, 'zero'); from compact import decode_torch
w = torch.load(sys.argv[1], map_location='cuda'); E, b = w['E'][:768].float(), w['b'].float()
idx = decode_torch(torch.from_numpy(np.load('data/ev_b0_c.npy', mmap_mode='r')[:200000].copy()).cuda()).long()
Mo = torch.zeros(len(idx), 769, device='cuda').scatter_add_(1, idx, torch.ones(idx.shape, device='cuda'))[:, :768]
a = b + Mo @ E; z = (a <= 0).float().mean().item(); o = (a >= 1).float().mean().item(); x = a.clamp(0, 1) ** 2
small = (x < 1 / 128).float().mean().item()
ch = (a.view(len(idx), -1, 16) <= 0).all(2).float().mean().item(); ch4 = (a.view(len(idx), -1, 4) <= 0).all(2).float().mean().item()
print(f'{sys.argv[1]}: width {E.shape[1]}; units at zero {z:.1%}, at one {o:.1%}, below 1/128 after squaring {small:.1%}; groups of 16 all zero {ch:.1%}, groups of 4 all zero {ch4:.1%}; mean activation {x.mean().item():.3f}')
