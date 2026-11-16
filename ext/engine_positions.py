"""Positions from our own engine's games, labelled by a deeper search of a stronger engine: the evaluator's test on the
positions that occur when engines play, as opposed to the human games it was trained on.

Self-play at a fixed number of positions per move from the match openings; kept are positions where the side to move is
not in check and the move played is neither a capture nor a promotion (the training data's filter), every third one.
Each is then searched by the labelling engine and its score turned into a win chance exactly as the training labels were.
Writes data/ev_<tag>_x.npy, data/ev_<tag>_y.npy (the trainer's format) and data/ev_<tag>.fen.

  python3 ext/engine_positions.py <tag> <games> <positions per move> <label positions per search> <workers> "<our engine> <network>" <labelling engine>
"""
import math, subprocess, sys
from multiprocessing import Pool
import chess, numpy as np
from match import openings

tag, games, nodes, lnodes, workers, us, lab = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
OPTS = 'Hash=16,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1'


class Eng:
    def __init__(self, cmd, opts=''):
        self.p = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.say('uci'); self.until('uciok')
        for o in opts.split(','):
            if o:
                self.say(f'setoption name {o.split("=")[0]} value {o.split("=")[1]}')
        self.say('isready'); self.until('readyok')

    def say(self, s):
        self.p.stdin.write(s + '\n')

    def until(self, word):
        last = ''
        while True:
            l = self.p.stdout.readline()
            if not l:
                raise RuntimeError('engine died')
            if l.startswith('info') and ' score ' in l:
                last = l
            if l.startswith(word):
                return l, last


def play(fen):
    e = Eng(us, OPTS); b = chess.Board(fen); moves = []; keep = []
    while not b.is_game_over(claim_draw=True) and len(moves) < 300:
        e.say(f'position fen {fen}' + (' moves ' + ' '.join(moves) if moves else '')); e.say(f'go nodes {nodes}')
        mv = chess.Move.from_uci(e.until('bestmove')[0].split()[1])
        if not b.is_check() and not b.is_capture(mv) and not mv.promotion and len(moves) >= 8:
            keep.append(b.fen())
        b.push(mv); moves.append(mv.uci())
    e.say('quit')
    return keep[::3]


def label(fens):
    e = Eng(lab, 'Hash=16,Threads=1'); out = []
    for f in fens:
        e.say(f'position fen {f}'); e.say(f'go nodes {lnodes}'); info = e.until('bestmove')[1].split()
        k = info.index('score'); out.append(None if info[k + 1] == 'mate' else max(-3000, min(3000, int(info[k + 2]))))
    e.say('quit')
    return out


def feats(f):
    b = chess.Board(f); v = []
    for sq, p in b.piece_map().items():
        v.append(((p.piece_type - 1) + (0 if p.color == b.turn else 6)) * 64 + (sq if b.turn else sq ^ 56))
    return sorted(v) + [768] * (32 - len(v))


if __name__ == '__main__':
    with Pool(workers) as pool:
        fens = [f for g in pool.imap_unordered(play, openings(games, 7)) for f in g]
        print(f'{len(fens)} positions from {games} games', flush=True)
        parts = [fens[i::workers] for i in range(workers)]; cps = pool.map(label, parts)
    rows = [(f, c) for part, cp in zip(parts, cps) for f, c in zip(part, cp) if c is not None]
    np.save(f'data/ev_{tag}_x.npy', np.array([feats(f) for f, _ in rows], dtype=np.int16))
    np.save(f'data/ev_{tag}_y.npy', np.array([1 / (1 + math.exp(-c / 271.58)) for _, c in rows], dtype=np.float32))
    open(f'data/ev_{tag}.fen', 'w').write(''.join(f'{f};{c}\n' for f, c in rows))
    print(f'{len(rows)} labelled positions (forced mates dropped) -> data/ev_{tag}_*.npy')
