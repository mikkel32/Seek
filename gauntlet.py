"""Clock games against other engines on equal terms: same machine, one thread each, the same hash size, the same
time control, every opening played with both colours. No adjudication by evaluation: games end by the rules of
chess, or on the clock.

  python3 gauntlet.py --us "calc/calc_t6 zero/test_net.bin" --opp "Fruit 2.1|/usr/games/fruit|Hash=64,OwnBook=false" \
                      --tc 10+0.1 --openings 100 --workers 40 --out zero/gauntlet.jsonl

Prints (and appends to --out) one JSON line: our wins, draws, losses, score, Elo difference with its 95% range, and how many
games each side lost on time.
"""
import argparse, json, math, os, subprocess, time
import chess
from multiprocessing import Pool
from match import openings, elo_from

GRACE = 0.05                      # seconds of lag a side may overstep before it loses on time
MAX_PLIES = 500


class Engine:
    def __init__(self, cmd, opts):
        self.p = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send('uci'); self.wait('uciok')
        for k, v in opts:
            self.send(f'setoption name {k} value {v}')
        self.send('isready'); self.wait('readyok')

    def send(self, line):
        self.p.stdin.write(line + '\n'); self.p.stdin.flush()

    def wait(self, word):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError('engine process died')
            if line.startswith(word):
                return line

    def new_game(self):
        self.send('ucinewgame'); self.send('isready'); self.wait('readyok')

    def go(self, fen, moves, wt, bt, inc, nodes=0):
        self.send('position fen ' + fen + (' moves ' + ' '.join(moves) if moves else ''))
        if nodes:                                   # both sides search the same number of positions per move: no clock, no speed advantage
            self.send(f'go nodes {nodes}')
            return self.wait('bestmove').split()[1]
        self.send(f'go wtime {max(1, int(wt * 1000))} btime {max(1, int(bt * 1000))} winc {int(inc * 1000)} binc {int(inc * 1000)}')
        return self.wait('bestmove').split()[1]

    def quit(self):
        try:
            self.send('quit'); self.p.wait(timeout=2)
        except Exception:
            self.p.kill()


def parse_opts(s):
    return [tuple(kv.split('=', 1)) for kv in s.split(',') if kv]


OPP_NODES = 0   # --opp_nodes: the opponent's own budget per move when it differs from ours (set before the workers fork)


def one_game(us, opp, fen, us_white, base, inc, nodes):
    """-> (our score, plies, who lost on time or by an illegal move)"""
    us.new_game(); opp.new_game()
    board = chess.Board(fen); moves = []; clock = {chess.WHITE: base, chess.BLACK: base}; flag = None; res = None; ms = []
    while res is None:
        if board.is_checkmate():
            res = 0.0 if board.turn == chess.WHITE else 1.0                    # white's score
        elif board.is_stalemate() or board.is_insufficient_material() or board.halfmove_clock >= 100 or board.is_repetition(3) or board.ply() >= MAX_PLIES:
            res = 0.5
        else:
            eng = us if (board.turn == chess.WHITE) == us_white else opp
            t0 = time.perf_counter()
            mv = eng.go(fen, moves, clock[chess.WHITE], clock[chess.BLACK], inc, nodes if eng is us or not OPP_NODES else OPP_NODES)
            dt = time.perf_counter() - t0; ms.append(int(1000 * dt)); clock[board.turn] -= 0 if nodes else dt
            try:
                m = chess.Move.from_uci(mv); legal = m in board.legal_moves
            except Exception:
                legal = False
            if clock[board.turn] < -GRACE or not legal:                          # out of time (or an illegal move): the side to move loses
                res = 0.0 if board.turn == chess.WHITE else 1.0
                flag = ('us' if eng is us else 'opp') + ('_time' if legal else '_illegal')
            else:
                clock[board.turn] += inc; board.push(m); moves.append(mv)
    if os.environ.get('GAMES_LOG'):                          # GAMES_LOG=<file>: one line a game, to look at where games are lost
        import fcntl, json
        with open(os.environ['GAMES_LOG'], 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps({'fen': fen, 'us_white': us_white, 'score': res if us_white else 1.0 - res, 'flag': flag, 'moves': moves, 'ms': ms}) + '\n')
    return (res if us_white else 1.0 - res, board.ply(), flag)


def play(args):
    us_cmd, us_opts, opp_cmd, opp_opts, base, inc, games, nodes = args
    us, opp = Engine(us_cmd, parse_opts(us_opts)), Engine(opp_cmd, parse_opts(opp_opts))
    out = [one_game(us, opp, fen, us_white, base, inc, nodes) for fen, us_white in games]
    us.quit(); opp.quit()
    return out


# ---- sequential testing: one opening (both colours) per task, so a match can stop as soon as its answer is clear
_cfg = None; _eng = None


def _init(cfg):
    global _cfg
    _cfg = cfg


def play_pair(fen):
    global _eng
    us_cmd, us_opts, opp_cmd, opp_opts, base, inc, nodes = _cfg
    if _eng is None:
        _eng = (Engine(us_cmd, parse_opts(us_opts)), Engine(opp_cmd, parse_opts(opp_opts)))
    return [one_game(_eng[0], _eng[1], fen, w, base, inc, nodes) for w in (True, False)]


def llr(pairs, e0, e1):
    """log-likelihood ratio of "the difference is e1 Elo" against "it is e0", from the scores of opening pairs (each 0..2).
    Pairs, not games: the two games of an opening are not independent."""
    n = len(pairs); x = [t / 2 for t in pairs]; m = sum(x) / n; v = sum((t - m) ** 2 for t in x) / n
    if v <= 0:
        return 0.0
    s0, s1 = (1 / (1 + 10 ** (-e / 400)) for e in (e0, e1))
    return n * (s1 - s0) * (2 * m - s0 - s1) / (2 * v)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--us', required=True); ap.add_argument('--us_opts', default='Hash=64')
    ap.add_argument('--opp', required=True)                     # name|command|Option=value,...
    ap.add_argument('--tc', default='10+0.1'); ap.add_argument('--openings', type=int, default=100)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--seed', type=int, default=7); ap.add_argument('--out', default='')
    ap.add_argument('--nodes', type=int, default=0)             # > 0: no clock, both engines search this many positions per move
    ap.add_argument('--opp_nodes', type=int, default=0)         # with --nodes: the opponent searches this many instead
    ap.add_argument('--stop', default='')                       # "e0,e1": stop as soon as the games decide between a difference of e0 and of e1 Elo
                                                                # (5% error each way); --openings is then the most that will be played
    a = ap.parse_args()
    OPP_NODES = a.opp_nodes
    name, cmd, opts = (a.opp.split('|') + [''])[:3]
    base, inc = (float(x) for x in a.tc.split('+'))
    games = [(f, w) for f in openings(a.openings, a.seed) for w in (True, False)]
    chunks = [games[i::a.workers] for i in range(a.workers)]
    t0 = time.time(); verdict = None
    if a.stop:
        e0, e1 = (float(x) for x in a.stop.split(',')); bound = math.log(0.95 / 0.05); res = []; pairs = []; L = 0.0
        pool = Pool(a.workers, initializer=_init, initargs=((a.us, a.us_opts, cmd, opts, base, inc, a.nodes),))
        for two in pool.imap_unordered(play_pair, openings(a.openings, a.seed)):
            res += two; pairs.append(two[0][0] + two[1][0])
            if len(pairs) >= 40:
                L = llr(pairs, e0, e1)
                if abs(L) >= bound:
                    verdict = f'decided after {len(res)} games: {"at least" if L > 0 else "not"} +{e1:g} rather than {"+" if L > 0 else ""}{e0:g}'; break
        pool.terminate(); pool.join()
        verdict = verdict or f'undecided between {e0:g} and {e1:g} after all {len(res)} games (log-likelihood ratio {L:+.2f}, decision at +-{bound:.2f})'
    else:
        with Pool(a.workers) as pool:
            res = [r for part in pool.map(play, [(a.us, a.us_opts, cmd, opts, base, inc, c, a.nodes) for c in chunks if c]) for r in part]
    w = sum(r[0] == 1.0 for r in res); d = sum(r[0] == 0.5 for r in res); l = sum(r[0] == 0.0 for r in res)
    e = elo_from(w, d, l)
    out = dict(opponent=name, tc=(f'{a.nodes} positions per move' + (f', the opponent {a.opp_nodes}' if a.opp_nodes else '')) if a.nodes else a.tc, **e, avg_plies=round(sum(r[1] for r in res) / len(res), 1),
               our_time_losses=sum(r[2] == 'us_time' for r in res), their_time_losses=sum(r[2] == 'opp_time' for r in res),
               illegal=sum(bool(r[2]) and r[2].endswith('illegal') for r in res), minutes=round((time.time() - t0) / 60, 1))
    if verdict:
        out['verdict'] = verdict
    line = json.dumps(out); print(line)
    if a.out:
        open(a.out, 'a').write(line + '\n')
