"""Find the network embedded in the installed Stockfish binary and write it out (for a private diagnostic: Stockfish's
evaluator inside our search). Prints the header's architecture description."""
import struct, sys
b = open('/usr/games/stockfish', 'rb').read(); magic = struct.pack('<I', 0x7AF32F20); i = b.find(magic); hits = []
while i >= 0:
    hits.append(i); i = b.find(magic, i + 1)
print('version marker at', hits)
for i in hits:
    ver, hsh, n = struct.unpack('<III', b[i:i + 12])
    if 0 < n < 1000:
        desc = b[i + 12:i + 12 + n]; print('hash %08x, description (%d bytes): %s' % (hsh, n, desc.decode('latin1')))
        open(sys.argv[1], 'wb').write(b[i:]); print('written from offset', i, 'length', len(b) - i)
        break
