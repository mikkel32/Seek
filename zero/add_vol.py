"""Attach the reliability read-out fitted by zero/vol_probe.py to an engine network file.

The read-out predicts, from the evaluator's hidden layer, how far a search is likely to move the static score (in score
units). The engine uses it to scale its pruning margins (calc_s5: options VolMix, VolRef). It is a trailer after the
network: tag 'VOL1', then for every piece-count group the weights for the mover's hidden units, the opponent's, and a bias.

  python3 zero/add_vol.py <network .bin> <zero/vol_head.pt> <out .bin>
"""
import struct, sys
import numpy as np, torch
src, head, dst = sys.argv[1:4]
d = open(src, 'rb').read(); magic, H, nb = struct.unpack('<iii', d[:12])
W = torch.load(head, map_location='cpu')['W'].numpy().astype('<f4')
assert W.shape == (nb, 2 * H + 1), (W.shape, nb, H)
print(f'{src}: hidden {H}, {nb} groups; read-out weights: largest {np.abs(W[:, :-1]).max():.2f}, 99.9th percentile {np.quantile(np.abs(W[:, :-1]), 0.999):.2f}, biases {W[:, -1].min():.2f} to {W[:, -1].max():.2f}')
open(dst, 'wb').write(d + struct.pack('<i', 0x314C4F56) + W.tobytes())
