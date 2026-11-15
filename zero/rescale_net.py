"""Write a copy of a network file whose evaluation is divided by k (python3 zero/rescale_net.py <in .bin> <out .bin> <k>).
The output side of the network is linear, so scaling its weights scales the evaluation exactly; the search's margins are in
centipawns, so two networks should be compared on the same scale."""
import struct, sys, numpy as np
src, dst, k = sys.argv[1], sys.argv[2], float(sys.argv[3])
b = open(src, 'rb').read(); magic, H, NB = struct.unpack('<iii', b[:12]); assert magic == 0x4F555232, hex(magic)
off = 12 + (768 * H + H) * 4                                   # header, input weights, biases: untouched
n_out = NB * 2 * H + 1 + 768 + 1 + 768 + NB                    # heads, the two tables with their constants, head constants
out = np.frombuffer(b[off:off + n_out * 4], dtype='<f4') / k
open(dst, 'wb').write(b[:off] + out.astype('<f4').tobytes() + b[off + n_out * 4:])
print(f'{src} -> {dst}: evaluation divided by {k:.3f} (hidden {H}, {NB} heads)')
