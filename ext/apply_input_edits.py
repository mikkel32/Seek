"""Apply or undo a Seek first-layer weight patch with byte-exact integrity checks."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile


def apply(source, manifest, destination, reverse=False):
    patch = json.loads(manifest.read_text())
    if isinstance(patch, list):
        # The fitter stores checkpoint hashes and tensor dimensions alongside the edits.
        summary = json.loads(manifest.with_name('summary.json').read_text())
        patch = {**summary, 'edits': patch}
    before, after = ('candidate', 'parent') if reverse else ('parent', 'candidate')
    data = bytearray(source.read_bytes())
    if hashlib.sha256(data).hexdigest() != patch[before + '_sha256']:
        raise ValueError('Source SHA-256 does not match the patch')
    magic, width, _ = struct.unpack_from('<III', data)
    if magic not in (0x4F555232, 0x4F555233):
        raise ValueError('Unsupported network format')
    offset = 16 if magic == 0x4F555233 else 12
    banks = struct.unpack_from('<I', data, 12)[0] if offset == 16 else 1
    rows = 768 * banks
    if list(patch['shape'][:2]) != [width, rows] or len(data) < offset + width * rows * 4:
        raise ValueError('Patch dimensions differ from the input layer')
    old_key, new_key = ('after', 'before') if reverse else ('before', 'after')
    used = set()
    for edit in patch['edits']:
        row, unit = edit['row'], edit['unit']
        if (type(row) is not int or type(unit) is not int or not 0 <= row < rows or
                not 0 <= unit < width or (row, unit) in used):
            raise ValueError('Repeated or invalid input-layer coordinate')
        used.add((row, unit))
        i = offset + 4 * (row * width + unit)
        old, new = edit[old_key], edit[new_key]
        if not math.isfinite(old) or not math.isfinite(new) or data[i:i+4] != struct.pack('<f', old):
            raise ValueError(f'Weight mismatch at row {row}, unit {unit}')
        struct.pack_into('<f', data, i, new)
    digest = hashlib.sha256(data).hexdigest()
    if digest != patch[after + '_sha256']:
        raise ValueError('Output SHA-256 does not match the patch')
    with tempfile.NamedTemporaryFile(dir=destination.parent) as out:
        out.write(data)
        out.flush()
        os.link(out.name, destination)
    return digest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('source', 'manifest', 'destination'):
        parser.add_argument(arg, type=Path)
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    print(apply(args.source, args.manifest, args.destination, args.reverse))
