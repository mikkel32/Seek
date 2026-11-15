"""The engine tunes its search FOR THE CLOCK by playing itself (SPSA). No other engine is involved.

The first self-tuning run (zero/spsa.py) gave both sides the same number of positions per move; its result was worth
+50 Elo at a fixed number of positions and nothing at the clock: with positions free, the tuner buys accuracy that costs
time. Pruning and reduction margins have to be judged where they are used, with the clock running.

Each iteration perturbs every parameter up and down at once, plays one opening with both colours between the two
variants at a real time control (the same clock rules as gauntlet.py), and moves every parameter a little towards the
variant that scored better. Iterations run in parallel and each one starts from the newest values.

  ENGINE=calc/calc_s4 python3 zero/spsa_clock.py <network> <iterations> [time control=5+0.05]
  resumable: the state is in zero/spsa_clock.json (SPSA_STATE=... for another file); SPSA_PROCS games at once (48)
"""
import json, os, random, sys, time
import chess
from multiprocessing import Pool

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, '.')
from gauntlet import Engine, GRACE, MAX_PLIES
from match import openings

BIN = os.environ.get('ENGINE', 'calc/calc_s4'); PROCS = int(os.environ.get('SPSA_PROCS', 48)); STATE = os.environ.get('SPSA_STATE', 'zero/spsa_clock.json')
#        name        start   min    max   perturbation at the end of the run
PARAMS = [('Razor', 293, 80, 700, 30), ('Rfp', 53, 15, 200, 8), ('NmpBase', 3, 1, 6, 0.5), ('NmpDiv', 4, 2, 8, 0.5), ('NmpEv', 191, 60, 600, 25),
          ('LmpBase', 6, 1, 14, 1), ('FutBase', 89, 0, 300, 15), ('FutMul', 94, 20, 250, 10), ('HistPrune', 4245, 500, 12000, 500),
          ('SeeQ', 48, 10, 200, 8), ('SeeC', 92, 20, 300, 12), ('SeDepth', 7, 4, 10, 0.5), ('SeMul', 2, 1, 6, 0.5), ('Asp', 33, 6, 80, 4),
          ('DeltaQ', 190, 50, 600, 25), ('LmrBase', 68, 10, 200, 10), ('LmrDiv', 240, 120, 400, 15), ('HbQuad', 5, 1, 16, 1), ('HbLin', 33, 0, 120, 6),
          ('HbMax', 1322, 300, 4000, 150), ('LmrHist', 8192, 2000, 30000, 800), ('TmSoft', 55, 25, 110, 5)]
FIXED = os.environ.get('SPSA_FIXED', '')              # options both variants always get, e.g. "CutLmr=1,NegExt=1"
W = sys.argv[1]; N = int(sys.argv[2]); TC = sys.argv[3] if len(sys.argv) > 3 else '5+0.05'
BASE, INC = (float(x) for x in TC.split('+'))
ALPHA, GAMMA, R_END = 0.602, 0.101, 0.002
A = 0.1 * N
_eng = None


def game(e_w, e_b, fen):
    """one clock game -> white's score"""
    e_w.new_game(); e_b.new_game()
    board = chess.Board(fen); moves = []; clock = {chess.WHITE: BASE, chess.BLACK: BASE}
    while True:
        if board.is_checkmate():
            return 0.0 if board.turn == chess.WHITE else 1.0
        if board.is_stalemate() or board.is_insufficient_material() or board.halfmove_clock >= 100 or board.is_repetition(3) or board.ply() >= MAX_PLIES:
            return 0.5
        eng = e_w if board.turn == chess.WHITE else e_b
        t0 = time.perf_counter(); mv = eng.go(fen, moves, clock[chess.WHITE], clock[chess.BLACK], INC); clock[board.turn] -= time.perf_counter() - t0
        try:
            m = chess.Move.from_uci(mv); legal = m in board.legal_moves
        except Exception:
            legal = False
        if clock[board.turn] < -GRACE or not legal:
            return 0.0 if board.turn == chess.WHITE else 1.0
        clock[board.turn] += INC; board.push(m); moves.append(mv)


def pair(args):
    global _eng
    plus, minus, fen = args
    try:
        if _eng is None:
            _eng = [Engine(f'{BIN} {W}', [('Hash', '64')] + [tuple(kv.split('=', 1)) for kv in FIXED.split(',') if kv]) for _ in range(2)]
        for e, th in zip(_eng, (plus, minus)):
            for k, v in th.items():
                e.send(f'setoption name {k} value {v}')
            e.send('isready'); e.wait('readyok')
        s = game(_eng[0], _eng[1], fen) + (1.0 - game(_eng[1], _eng[0], fen))        # the plus variant's points out of 2
        return 2.0 * s - 2.0                                                        # wins minus losses, in [-2, 2]
    except Exception:                                                               # an engine died: start fresh ones, count nothing
        for e in _eng or []:
            try:
                e.p.kill()
            except Exception:
                pass
        _eng = None
        return 0.0


if __name__ == '__main__':
    st = json.load(open(STATE)) if os.path.exists(STATE) else {'k': 0, 'theta': {p[0]: float(p[1]) for p in PARAMS}, 'log': []}
    theta = st['theta']; done = st['k']; issued = done; ops = openings(4000, 2024); rs = random.Random(done + 1)
    spec = {p[0]: p for p in PARAMS}
    clip = lambda n, v: min(spec[n][3], max(spec[n][2], v))
    t0 = time.time(); start = done; pending = []; score = 0.0
    print(f'{BIN} on {W}, {TC}, {PROCS} games at once, iterations {done} -> {N}' + (f', fixed options {FIXED}' if FIXED else ''), flush=True)
    with Pool(PROCS) as pool:
        while done < N and not os.path.exists('zero/SPSA_STOP'):
            while len(pending) < PROCS and issued < N:
                issued += 1; kk = issued
                ck = {n: spec[n][4] * (N / kk) ** GAMMA for n in theta}
                d = {n: rs.choice((-1, 1)) for n in theta}
                plus = {n: int(round(clip(n, theta[n] + ck[n] * d[n]))) for n in theta}
                minus = {n: int(round(clip(n, theta[n] - ck[n] * d[n]))) for n in theta}
                pending.append((kk, ck, d, pool.apply_async(pair, ((plus, minus, ops[kk % len(ops)]),))))
            time.sleep(0.05)
            for item in [p for p in pending if p[3].ready()]:
                kk, ck, d, ar = item; r = ar.get(); pending.remove(item); done += 1; score += abs(r)
                for n in theta:
                    a_end = R_END * spec[n][4] ** 2
                    ak = a_end * (A + N) ** ALPHA / (A + kk) ** ALPHA
                    theta[n] = clip(n, theta[n] + ak / ck[n] * r * d[n])
                if done % 250 == 0:
                    st = {'k': done, 'theta': theta, 'log': st.get('log', []) + [[done, {n: round(v, 2) for n, v in theta.items()}]]}
                    json.dump(st, open(STATE, 'w'))
                    print(f'[{time.strftime("%H:%M:%S", time.gmtime())}] iteration {done}/{N}  {(done - start) * 2 / (time.time() - t0):.1f} games/s  '
                          + ' '.join(f'{n}={theta[n]:.0f}' for n in theta), flush=True)
    json.dump({'k': done, 'theta': theta, 'log': st.get('log', [])}, open(STATE, 'w'))
    print('options: ' + ','.join(f'{n}={int(round(v))}' for n, v in theta.items()))
