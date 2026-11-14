"""The engine tunes its own search by playing itself (SPSA). No other engine is involved.

Each iteration perturbs all parameters up and down at once, plays one opening with both colours between the two
variants at an equal node budget, and moves every parameter a little towards the variant that scored better.

  python3 zero/spsa.py <weights> <iterations> [nodes=5000]      (resumable: state in zero/spsa.json)
"""
import json, math, os, random, sys, time
import chess, chess.engine
from multiprocessing import Pool

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
BIN = os.environ.get('ENGINE', 'calc/calc_f4'); PROCS = int(os.environ.get('SPSA_PROCS', 3))
#        name          default  min   max    step at the end of the run
PARAMS = [('Razor', 250, 80, 600, 30), ('Rfp', 70, 20, 200, 10), ('NmpEv', 200, 60, 600, 30), ('LmpBase', 3, 1, 12, 1),
          ('FutBase', 90, 0, 300, 15), ('FutMul', 85, 20, 250, 10), ('HistPrune', 4000, 500, 12000, 500), ('SeeQ', 50, 10, 200, 8),
          ('SeeC', 90, 20, 300, 12), ('SeMul', 2, 1, 6, 0.5), ('Asp', 25, 6, 80, 4), ('DeltaQ', 200, 50, 600, 25),
          ('LmrBase', 75, 10, 200, 10), ('LmrDiv', 225, 120, 400, 15), ('HbQuad', 4, 1, 16, 1), ('HbLin', 20, 0, 120, 6), ('HbMax', 1200, 300, 4000, 150)]
W = sys.argv[1]; N = int(sys.argv[2]); NODES = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
ALPHA, GAMMA, R_END = 0.602, 0.101, 0.002
A = 0.1 * N
_eng = None


def openings():
    """balanced-ish openings made by the engine's own rules: 8 random plies from the start position"""
    rs = random.Random(12345); out = []
    while len(out) < 4000:
        b = chess.Board()
        for _ in range(8):
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(rs.choice(ms))
        if not b.is_game_over():
            out.append(b.fen())
    return out


def game(e_w, e_b, fen):
    b = chess.Board(fen)
    while True:
        if b.is_checkmate():
            return 0.0 if b.turn == chess.WHITE else 1.0
        if b.is_stalemate() or b.is_insufficient_material() or b.halfmove_clock >= 100 or b.is_repetition(3) or b.ply() >= 300:
            return 0.5
        b.push((e_w if b.turn == chess.WHITE else e_b).play(b, chess.engine.Limit(nodes=NODES)).move)


def pair(args):
    global _eng
    plus, minus, fen = args
    if _eng is None:
        _eng = [chess.engine.SimpleEngine.popen_uci([BIN, W]) for _ in range(2)]
    _eng[0].configure(plus); _eng[1].configure(minus)
    s = game(_eng[0], _eng[1], fen) + (1.0 - game(_eng[1], _eng[0], fen))        # plus's points out of 2
    return 2.0 * s - 2.0                                                        # wins minus losses, in [-2, 2]


if __name__ == '__main__':
    st = json.load(open('zero/spsa.json')) if os.path.exists('zero/spsa.json') else {'k': 0, 'theta': {p[0]: float(p[1]) for p in PARAMS}}
    theta = st['theta']; k0 = st['k']; ops = openings(); rs = random.Random(k0 + 1)
    spec = {p[0]: p for p in PARAMS}
    clip = lambda n, v: min(spec[n][3], max(spec[n][2], v))
    t0 = time.time()
    with Pool(PROCS) as pool:
        k = k0
        while k < N:
            batch = []
            for j in range(PROCS):
                kk = k + j + 1
                ck = {n: spec[n][4] * (N / kk) ** GAMMA for n in theta}
                d = {n: rs.choice((-1, 1)) for n in theta}
                plus = {n: int(round(clip(n, theta[n] + ck[n] * d[n]))) for n in theta}
                minus = {n: int(round(clip(n, theta[n] - ck[n] * d[n]))) for n in theta}
                batch.append((kk, ck, d, (plus, minus, ops[kk % len(ops)])))
            res = pool.map(pair, [b[3] for b in batch])
            for (kk, ck, d, _), r in zip(batch, res):
                for n in theta:
                    a_end = R_END * spec[n][4] ** 2
                    ak = a_end * (A + N) ** ALPHA / (A + kk) ** ALPHA
                    theta[n] = clip(n, theta[n] + ak / ck[n] ** 2 * ck[n] * r * d[n])
            k += PROCS
            if k // 200 != (k - PROCS) // 200:
                json.dump({'k': k, 'theta': theta}, open('zero/spsa.json', 'w'))
                print(f'iteration {k}/{N}  {(k - k0) * 2 / (time.time() - t0):.1f} games/s  ' + ' '.join(f'{n}={theta[n]:.0f}' for n in theta), flush=True)
    json.dump({'k': k, 'theta': theta}, open('zero/spsa.json', 'w'))
    print('final', {n: int(round(v)) for n, v in theta.items()})
