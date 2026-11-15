"""Engine network file (.bin) -> a trainer checkpoint (.pt) with every weight, so that training can continue from a network of
which only the engine file exists (calc/train_nnue.py --widen), and a table file (.npz) with its piece values.

  python3 zero/bin2pt.py <network .bin> <out .pt> [<out .npz>]      (plain 768-input networks, with or without trailers)
"""
import struct, sys
import numpy as np, torch
d = open(sys.argv[1], 'rb').read(); magic, H, nb = struct.unpack('<iii', d[:12]); assert magic == 0x4F555232, hex(magic)
o = 12
def take(n):
    global o
    a = np.frombuffer(d, '<f4', n, o).copy(); o += 4 * n
    return a
E = take(768 * H).reshape(768, H); b = take(H); wo = np.zeros((nb, H), np.float32); wp = np.zeros((nb, H), np.float32)
for k in range(nb):
    wo[k] = take(H); wp[k] = take(H)
c0 = take(1); S0 = take(768); c1 = take(1); S1 = take(768); cb = take(nb)
t = torch.from_numpy; z = lambda *s: torch.zeros(*s)
torch.save({'E': torch.cat([t(E), z(1, H)]), 'b': t(b), 'wo': t(wo), 'wp': t(wp), 'cb': t(cb), 'S': torch.cat([torch.stack([t(S0), t(S1)], 1), z(1, 2)]), 'c': torch.tensor([float(c0[0]), float(c1[0])])}, sys.argv[2])
pv = None
if o + 24 <= len(d) and struct.unpack('<i', d[o:o + 4])[0] == 0x4C415650:
    pv = list(struct.unpack('<5i', d[o + 4:o + 24]))
if len(sys.argv) > 3:
    np.savez(sys.argv[3], pval=np.array(pv if pv else [100, 320, 330, 500, 950]), S_mg=S0, S_eg=S1, c_mg=c0[0], c_eg=c1[0])
print(f'{sys.argv[1]}: hidden {H}, {nb} output groups, piece values {pv} -> {sys.argv[2]}')
