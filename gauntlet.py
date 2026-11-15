"""Clock games against other engines on equal terms: same machine, one thread each, the same hash size, the same
time control, every opening played with both colours. No adjudication by evaluation: games end by the rules of
chess, or on the clock.

  python3 gauntlet.py --us "calc/calc_t6 zero/test_net.bin" --opp "Fruit 2.1|/usr/games/fruit|Hash=64,OwnBook=false" \
                      --tc 10+0.1 --openings 100 --workers 40 --out zero/gauntlet.jsonl

Prints (and appends to --out) one JSON line: our wins, draws, losses, score, Elo difference with its 95% range, and how many
games each side lost on time.
"""
import argparse, json, math, subprocess, time
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

    def go(self, fen, moves, wt, bt, inc):
        self.send('position fen ' + fen + (' moves ' + ' '.join(moves) if moves else ''))
        self.send(f'go wtime {max(1, int(wt * 1000))} btime {max(1, int(bt * 1000))} winc {int(inc * 1000)} binc {int(inc * 1000)}')
        return self.wait('bestmove').split()[1]

    def quit(self):
        try:
            self.send('quit'); self.p.wait(timeout=2)
        except Exception:
            self.p.kill()


def parse_opts(s):
    return [tuple(kv.split('=', 1)) for kv in s.split(',') if kv]


def play(args):
    us_cmd, us_opts, opp_cmd, opp_opts, base, inc, games = args
    us, opp = Engine(us_cmd, parse_opts(us_opts)), Engine(opp_cmd, parse_opts(opp_opts))
    out = []
    for fen, us_white in games:
        us.new_game(); opp.new_game()
        board = chess.Board(fen); moves = []; clock = {chess.WHITE: base, chess.BLACK: base}; flag = None; res = None
        while res is None:
            if board.is_checkmate():
                res = 0.0 if board.turn == chess.WHITE else 1.0                    # white's score
            elif board.is_stalemate() or board.is_insufficient_material() or board.halfmove_clock >= 100 or board.is_repetition(3) or board.ply() >= MAX_PLIES:
                res = 0.5
            else:
                eng = us if (board.turn == chess.WHITE) == us_white else opp
                t0 = time.perf_counter()
                mv = eng.go(fen, moves, clock[chess.WHITE], clock[chess.BLACK], inc)
                clock[board.turn] -= time.perf_counter() - t0
                try:
                    m = chess.Move.from_uci(mv); legal = m in board.legal_moves
                except Exception:
                    legal = False
                if clock[board.turn] < -GRACE or not legal:                          # out of time (or an illegal move): the side to move loses
                    res = 0.0 if board.turn == chess.WHITE else 1.0
                    flag = ('us' if eng is us else 'opp') + ('_time' if legal else '_illegal')
                else:
                    clock[board.turn] += inc; board.push(m); moves.append(mv)
        out.append((res if us_white else 1.0 - res, board.ply(), flag))
    us.quit(); opp.quit()
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--us', required=True); ap.add_argument('--us_opts', default='Hash=64')
    ap.add_argument('--opp', required=True)                     # name|command|Option=value,...
    ap.add_argument('--tc', default='10+0.1'); ap.add_argument('--openings', type=int, default=100)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--seed', type=int, default=7); ap.add_argument('--out', default='')
    a = ap.parse_args()
    name, cmd, opts = (a.opp.split('|') + [''])[:3]
    base, inc = (float(x) for x in a.tc.split('+'))
    games = [(f, w) for f in openings(a.openings, a.seed) for w in (True, False)]
    chunks = [games[i::a.workers] for i in range(a.workers)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = [r for part in pool.map(play, [(a.us, a.us_opts, cmd, opts, base, inc, c) for c in chunks if c]) for r in part]
    w = sum(r[0] == 1.0 for r in res); d = sum(r[0] == 0.5 for r in res); l = sum(r[0] == 0.0 for r in res)
    e = elo_from(w, d, l)
    out = dict(opponent=name, tc=a.tc, **e, avg_plies=round(sum(r[1] for r in res) / len(res), 1),
               our_time_losses=sum(r[2] == 'us_time' for r in res), their_time_losses=sum(r[2] == 'opp_time' for r in res),
               illegal=sum(bool(r[2]) and r[2].endswith('illegal') for r in res), minutes=round((time.time() - t0) / 60, 1))
    line = json.dumps(out); print(line)
    if a.out:
        open(a.out, 'a').write(line + '\n')
