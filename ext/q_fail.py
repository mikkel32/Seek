"""Which rule hides the moves we miss?  (data of ext/quality.py)

The failure set: one-good-move positions that Stockfish solves at a small budget and our engine still misses at a large
one. Each variant (a set of engine options, typically one search rule switched off with the Off option) searches these
positions with the large budget; printed is the share it solves. A rule whose removal solves many of them, although the
search then reaches less depth, is the one throwing those moves away.

  python3 ext/q_fail.py --sf sf_100000 --ours bigm_1000000 --cmd "calc/calc_s19h zero/w_BIGM.bin" --base "Hash=64,..." --nodes 1000000 "name|Options" ...
"""
import argparse, json, math, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quality as Q

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--sf'); ap.add_argument('--ours'); ap.add_argument('--cmd'); ap.add_argument('--base', default='Hash=64')
    ap.add_argument('--nodes', type=int, default=1000000); ap.add_argument('--workers', type=int, default=22); ap.add_argument('variants', nargs='*')
    a = ap.parse_args()
    tr = {t['fen']: t for t in Q.usable() if t['cp2'] is not None and Q.E(t['cp1']) - Q.E(t['cp2']) >= 0.04}
    rs, ro = (json.load(open(f'{Q.D}/runs/{x}.json')) for x in (a.sf, a.ours)); ms, mo = dict(zip(rs['fens'], rs['moves'])), dict(zip(ro['fens'], ro['moves']))
    fail = [f for f in tr if f in ms and f in mo and ms[f] == tr[f]['m1'] and mo[f] != tr[f]['m1']]
    both = sum(1 for f in tr if f in ms and f in mo)
    print(f'{len(fail)} of {both} one-good-move positions: {a.sf} finds the move, {a.ours} does not', flush=True)
    items = list(enumerate(fail)); chunks = [items[k::a.workers] for k in range(a.workers)]
    for v in a.variants:
        name, opts = (v.split('|') + [''])[:2]; nodes = a.nodes
        if opts.startswith('nodes='):
            nodes = int(opts.split(',')[0][6:]); opts = ','.join(opts.split(',')[1:])
        with Pool(a.workers) as pool:
            res = [x for r in pool.map(Q.w_test, [(a.cmd, a.base + (',' + opts if opts else ''), nodes, c) for c in chunks if c]) for x in r]
        ok = sum(1 for i, mv in res if mv == tr[fail[i]]['m1'])
        print(f'{name:28s} {nodes:>9d} nodes: solves {ok:4d} = {100 * ok / len(fail):5.1f}%   [{opts}]', flush=True)
