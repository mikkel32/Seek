"""Build the two tested BIGS candidates from descendants trained from the same initialization."""
import argparse
import os
from pathlib import Path
import struct
import tempfile

import numpy as np


def blend(parent, control, sibling, output, recipe):
    header = struct.pack("<4i", 0x4F555233, 768, 8, 16)
    num_weights = 16 * 768 * 768 + 768 + 8 * 2 * 768 + 2 * (1 + 768) + 8
    end = len(header) + num_weights * 4

    def load(path):
        data = path.read_bytes()
        tail = data[end:]
        if data[:16] != header or len(data) < end:
            raise ValueError(f"{path} is not a complete 768-wide BIGS evaluator")
        if tail and (len(tail) != 24 or tail[:4] != struct.pack("<I", 0x4C415650)):
            raise ValueError(f"Unknown auxiliary data in {path}")
        floats = np.frombuffer(data, "<f4", count=num_weights, offset=16)
        if not np.isfinite(floats).all():
            raise ValueError(f"Non-finite weight in {path}")
        return floats, tail

    base, parent_trailer = load(parent)
    control_factor, sibling_factor = {"average": (0.5, 0.25), "contrast": (0.75, -0.25)}[recipe]
    result = base.copy()
    for checkpoint, factor in [(control, control_factor), (sibling, sibling_factor)]:
        weights, _ = load(checkpoint)
        result += (weights - base) * np.float32(factor)
    with tempfile.NamedTemporaryFile(dir=output.parent) as handle:
        handle.write(header)
        handle.write(result.tobytes())
        handle.write(parent_trailer)
        handle.flush()
        os.link(handle.name, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", choices=("average", "contrast"))
    parser.add_argument("parent", type=Path, help="original w_BIGS.bin")
    parser.add_argument("control", type=Path, help="w_C1a.bin")
    parser.add_argument("sibling", type=Path, help="w_G1a.bin for average, w_G1b.bin for contrast")
    parser.add_argument("output", type=Path, help="new path for the blended network")
    args = parser.parse_args()
    blend(args.parent, args.control, args.sibling, args.output, args.recipe)
