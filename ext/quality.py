"""Search quality at a fixed node budget: how much does an engine give away per move, judged by a much deeper search?

Games are a slow instrument for deep searches (a game at a million nodes a move takes minutes and decides one point).
Here every position is a sample: the engine under test picks a move with a fixed node budget, and the move is scored
against Stockfish's own deeper two-line analysis of the same position. The engine is deterministic at a node budget, so
two configurations differ only where they really choose differently, and the comparison is paired.

  collect   games between two engines at a node budget; a few positions of every game are kept (full FEN)
  truth     Stockfish at a large node budget, two lines: best move, its score, the second move, its score
  test      an engine configuration picks a move in every position; moves outside Stockfish's two lines are
            scored by Stockfish searching that move alone (cached)
  cmp       two tested configurations, position by position

  python3 ext/quality.py collect --us "calc/calc_s19c zero/w_BIGM.bin" --us_opts Hash=64 --opp /usr/games/stockfish --nodes 30000 --games 1000
  python3 ext/quality.py truth --nodes 4000000
  python3 ext/quality.py test --name bigm_300k --cmd "calc/calc_s19c zero/w_BIGM.bin" --opts Hash=64,PvPrune=1 --nodes 300000
  python3 ext/quality.py cmp bigm_300k other_300k

Loss of a move = expected score of Stockfish's best move minus expected score of the move played, both from Stockfish's
scores through 1 / (1 + exp(-centipawns / 200)). Reported: mean loss in thousandths of a point per move, the share of
moves losing 0.05 or more, and the share of "one good move" positions (second line at least 0.04 worse) solved.
Files live in data/q/.
"""
import argparse, json, math, os, random, subprocess, sys, time
from multiprocessing import Pool
import chess

D = 'data/q'; SF = '/usr/games/stockfish'


class Eng:
    def __init__(self, cmd, opts):
        argv = cmd.split() if isinstance(cmd, str) else cmd
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send('uci'); self.wait('uciok')
        for k, v in opts:
            self.send(f'setoption name {k} value {v}')
        self.send('isready'); self.wait('readyok')

    def send(self, s):
        self.p.stdin.write(s + '\n'); self.p.stdin.flush()

    def wait(self, word, keep=None):
        while True:
            l = self.p.stdout.readline()
            if not l:
                raise RuntimeError('engine process died')
            if keep is not None and l.startswith('info') and ' score ' in l and ' pv ' in l:
                keep.append(l)
            if l.startswith(word):
                return l

    def new(self):
        self.send('ucinewgame'); self.send('isready'); self.wait('readyok')

    def move_time(self, fen, ms):
        self.send('position fen ' + fen); self.send(f'go movetime {ms}')
        return self.wait('bestmove').split()[1]

    def move(self, fen, nodes, moves=None, extra=''):
        self.send('position fen ' + fen + (' moves ' + ' '.join(moves) if moves else ''))
        self.send(f'go nodes {nodes}{extra}')
        return self.wait('bestmove').split()[1]

    def lines(self, fen, nodes, extra=''):
        """-> {multipv index: (move, centipawns from the mover's side)} of the last depth reported"""
        self.send('position fen ' + fen); self.send(f'go nodes {nodes}{extra}'); got = []; self.wait('bestmove', got); out = {}
        for l in got:
            t = l.split(); k = int(t[t.index('multipv') + 1]) if 'multipv' in t else 1; i = t.index('score')
            if 'upperbound' in t or 'lowerbound' in t:
                continue
            cp = int(t[i + 2]) if t[i + 1] == 'cp' else (30000 - 100 * abs(int(t[i + 2]))) * (1 if int(t[i + 2]) > 0 else -1)
            out[k] = (t[t.index('pv') + 1], cp)
        return out

    def quit(self):
        try:
            self.send('quit'); self.p.wait(timeout=2)
        except Exception:
            self.p.kill()


def opts_of(s):
    return [tuple(kv.split('=', 1)) for kv in s.split(',') if kv]


def E(cp):
    return 1 / (1 + math.exp(-max(-4000, min(4000, cp)) / 200))


def load(name):
    return [json.loads(l) for l in open(f'{D}/{name}')] if os.path.exists(f'{D}/{name}') else []


# ------------------------------------------------------------------ collect
def w_collect(a):
    us_cmd, us_opts, opp_cmd, nodes, per, jobs, seed = a
    us, opp = Eng(us_cmd, opts_of(us_opts)), Eng(opp_cmd, [('Hash', '64'), ('Threads', '1')]); rnd = random.Random(seed); out = []
    for gi, fen, us_white in jobs:
        us.new(); opp.new(); b = chess.Board(fen); moves = []; seen = []
        while not (b.is_game_over(claim_draw=True) or b.ply() > 400):
            if len(moves) >= 6 and b.legal_moves.count() >= 3:
                seen.append((b.fen(), len(moves)))
            e = us if (b.turn == chess.WHITE) == us_white else opp
            mv = e.move(fen, nodes, moves); m = chess.Move.from_uci(mv)
            if m not in b.legal_moves:
                break
            b.push(m); moves.append(mv)
        for f, ply in rnd.sample(seen, min(per, len(seen))):
            out.append({'fen': f, 'game': gi, 'ply': ply})
    us.quit(); opp.quit()
    return out


def collect(a):
    sys.path.insert(0, '.'); from match import openings
    fens = openings(a.skip + a.games // 2 + 1)[a.skip:a.skip + a.games // 2]; jobs = [(2 * i + c, f, bool(c)) for i, f in enumerate(fens) for c in (0, 1)]
    parts = [jobs[w::a.workers] for w in range(a.workers)]
    with Pool(a.workers) as pool:
        res = pool.map(w_collect, [(a.us, a.us_opts, a.opp, a.nodes, a.per, p, 1000 + i) for i, p in enumerate(parts)])
    pos = [x for r in res for x in r]; seenf = set(); uniq = []
    for x in pos:
        k = ' '.join(x['fen'].split()[:4])
        if k not in seenf:
            seenf.add(k); uniq.append(x)
    random.Random(7).shuffle(uniq)
    with open(f'{D}/positions.jsonl', 'a') as f:
        for x in uniq:
            f.write(json.dumps(x) + '\n')
    print(f'{len(jobs)} games, {len(uniq)} distinct positions added; {len(load("positions.jsonl"))} in data/q/positions.jsonl', flush=True)


# ------------------------------------------------------------------ truth
def w_truth(a):
    nodes, items = a; e = Eng(SF, [('Hash', '128'), ('Threads', '1'), ('MultiPV', '2')]); out = []
    for i, fen in items:
        e.new(); ln = e.lines(fen, 50000)                    # a quick look first: positions already decided are not analysed deeply
        if 1 in ln and abs(ln[1][1]) > 600:
            out.append({'i': i, 'fen': fen, 'm1': ln[1][0], 'cp1': ln[1][1], 'm2': None, 'cp2': None, 'nodes': 50000}); continue
        e.new(); ln = e.lines(fen, nodes)
        if 1 in ln:
            out.append({'i': i, 'fen': fen, 'm1': ln[1][0], 'cp1': ln[1][1], 'm2': ln[2][0] if 2 in ln else None, 'cp2': ln[2][1] if 2 in ln else None, 'nodes': nodes})
    e.quit()
    return out


def truth(a):
    pos = load('positions.jsonl'); have = {t['fen'] for t in load('truth.jsonl')}
    todo = [(i, p['fen']) for i, p in enumerate(pos) if p['fen'] not in have][:a.limit or None]; t0 = time.time()
    if not todo:
        print('truth: nothing left', flush=True); return
    chunks = [todo[k:k + 4] for k in range(0, len(todo), 4)]
    with Pool(a.workers) as pool, open(f'{D}/truth.jsonl', 'a') as f:
        for n, r in enumerate(pool.imap_unordered(w_truth, [(a.nodes, c) for c in chunks])):
            for x in r:
                f.write(json.dumps(x) + '\n')
            f.flush()
            if n % 100 == 99:
                print(f'{4 * (n + 1)}/{len(todo)} positions, {time.time() - t0:.0f}s', flush=True)
    print(f'truth for {len(load("truth.jsonl"))} positions ({time.time() - t0:.0f}s)', flush=True)


# ------------------------------------------------------------------ test
def w_test(a):
    cmd, opts, nodes, items = a; e = Eng(cmd, opts_of(opts)); out = []
    for i, fen in items:
        e.new(); out.append((i, e.move(fen, nodes) if nodes > 0 else e.move_time(fen, -nodes)))     # a negative budget is milliseconds per move
    e.quit()
    return out


def w_demand(a):
    nodes, items = a; e = Eng(SF, [('Hash', '128'), ('Threads', '1')]); out = []
    for fen, mv in items:
        e.new(); ln = e.lines(fen, nodes, f' searchmoves {mv}')
        if 1 in ln:
            out.append((fen, mv, ln[1][1]))
    e.quit()
    return out


def scores(tr, moves, dem):
    """per position: loss, and whether it is a one-good-move position"""
    loss, crit = [], []
    for t, mv in zip(tr, moves):
        e1 = E(t['cp1']); e2 = E(t['cp2']) if t['cp2'] is not None else e1
        if mv == t['m1']:
            l = 0.0
        elif mv == t['m2']:
            l = e1 - e2
        else:
            cp = dem.get(t['fen'] + ' ' + mv)
            l = e1 - min(e2, E(cp)) if cp is not None else e1 - e2
        loss.append(max(0.0, l)); crit.append(e1 - e2 >= 0.04)
    return loss, crit


def usable():
    """positions with a truth that are still a game (the best line within 5 pawns)"""
    return [t for t in load('truth.jsonl') if abs(t['cp1']) <= 500 and t['nodes'] > 50000]


def test(a):
    tr = usable()[a.first:a.limit or None]; uni = 0
    if a.subset:                                             # only the positions where configurations have been seen to differ
        sub = json.load(open(f'{D}/{a.subset}.json')); keep = set(sub['fens']); uni = sub['universe']; tr = [t for t in tr if t['fen'] in keep]
    t0 = time.time(); items = [(i, t['fen']) for i, t in enumerate(tr)]
    chunks = [items[k:k + 8] for k in range(0, len(items), 8)]; moves = [None] * len(tr)
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(w_test, [(a.cmd, a.opts, a.nodes, c) for c in chunks]):
            for i, mv in r:
                moves[i] = mv
    t1 = time.time()
    dem = json.load(open(f'{D}/demand.json')) if os.path.exists(f'{D}/demand.json') else {}
    need = [(t['fen'], mv) for t, mv in zip(tr, moves) if mv not in (t['m1'], t['m2']) and (t['fen'] + ' ' + mv) not in dem]
    if need:
        dn = max(200000, max(t['nodes'] for t in tr) // 2); ch = [need[k:k + 4] for k in range(0, len(need), 4)]
        with Pool(a.workers) as pool:
            for r in pool.imap_unordered(w_demand, [(dn, c) for c in ch]):
                for fen, mv, cp in r:
                    dem[fen + ' ' + mv] = cp
        json.dump(dem, open(f'{D}/demand.json', 'w'))
    os.makedirs(f'{D}/runs', exist_ok=True)
    json.dump({'cmd': a.cmd, 'opts': a.opts, 'nodes': a.nodes, 'fens': [t['fen'] for t in tr], 'moves': moves, 'universe': uni}, open(f'{D}/runs/{a.name}.json', 'w'))
    loss, crit = scores(tr, moves, dem); n = len(loss); nc = sum(crit)
    print(f'{a.name}: {n} positions at {a.nodes} nodes ({t1 - t0:.0f}s, {len(need)} moves sent to Stockfish, {time.time() - t1:.0f}s): '
          f'mean loss {1000 * sum(loss) / n:.2f} thousandths of a point per move; {100 * sum(l >= 0.05 for l in loss) / n:.2f}% of moves lose 0.05 or more; '
          f'one-good-move positions solved {100 * sum(l == 0 for l, c in zip(loss, crit) if c) / nc:.1f}% of {nc}', flush=True)


def cmp(a):
    ra, rb = (json.load(open(f'{D}/runs/{x}.json')) for x in (a.a, a.b)); trd = {t['fen']: t for t in usable()}
    dem = json.load(open(f'{D}/demand.json')) if os.path.exists(f'{D}/demand.json') else {}
    mb = dict(zip(rb['fens'], rb['moves'])); fens = [f for f in ra['fens'] if f in mb and f in trd]; tr = [trd[f] for f in fens]
    ma = dict(zip(ra['fens'], ra['moves'])); la, crit = scores(tr, [ma[f] for f in fens], dem); lb, _ = scores(tr, [mb[f] for f in fens], dem)
    d = [x - y for x, y in zip(la, lb)]; n = len(d); m = sum(d) / n; sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5; se = sd / n ** 0.5
    diff = sum(x != y for x, y in zip(la, lb))
    uni = min(ra.get('universe') or 0, rb.get('universe') or 0)
    if uni and uni > n:                                      # tested on a subset: the other positions of the universe count as no difference
        m *= n / uni; se *= n / uni
    print(f'{a.a} against {a.b}: {n} positions; mean loss {1000 * sum(la) / n:.2f} against {1000 * sum(lb) / n:.2f} thousandths; '
          f'difference {1000 * m:+.2f} (standard error {1000 * se:.2f}, {m / se if se else 0:+.1f} of them; negative = the first gives away less); '
          f'{diff} positions scored differently' + (f'; scaled to the {uni} positions the subset was drawn from' if uni and uni > n else ''), flush=True)


def subset(a):
    """the positions where the named runs do not all choose the same move (the others tell configurations apart very rarely)"""
    runs = [json.load(open(f'{D}/runs/{x}.json')) for x in a.runs.split(',')]; common = set(runs[0]['fens'])
    for r in runs[1:]:
        common &= set(r['fens'])
    maps = [dict(zip(r['fens'], r['moves'])) for r in runs]
    fens = [f for f in runs[0]['fens'] if f in common and len({m[f] for m in maps}) > 1]
    json.dump({'fens': fens, 'universe': len(common), 'runs': a.runs}, open(f'{D}/{a.out}.json', 'w'))
    print(f'{len(fens)} of {len(common)} positions have more than one move among {len(runs)} runs -> {D}/{a.out}.json', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='what')
    c = sub.add_parser('collect'); c.add_argument('--us'); c.add_argument('--us_opts', default='Hash=64'); c.add_argument('--opp', default=SF)
    c.add_argument('--nodes', type=int, default=30000); c.add_argument('--games', type=int, default=1000); c.add_argument('--per', type=int, default=8); c.add_argument('--workers', type=int, default=22); c.add_argument('--skip', type=int, default=0)
    t = sub.add_parser('truth'); t.add_argument('--nodes', type=int, default=4000000); t.add_argument('--workers', type=int, default=22); t.add_argument('--limit', type=int, default=0)
    s = sub.add_parser('test'); s.add_argument('--name'); s.add_argument('--cmd'); s.add_argument('--opts', default='Hash=64'); s.add_argument('--nodes', type=int)       # negative: milliseconds per move instead (equal time, so whatever a configuration costs in time is charged; keep --workers at or below the cores)
    s.add_argument('--workers', type=int, default=22); s.add_argument('--limit', type=int, default=0); s.add_argument('--first', type=int, default=0); s.add_argument('--subset', default='')   # positions [first, limit) of the usable list
    m = sub.add_parser('cmp'); m.add_argument('a'); m.add_argument('b')
    u = sub.add_parser('subset'); u.add_argument('--runs'); u.add_argument('--out', default='contested')
    a = ap.parse_args(); os.makedirs(D, exist_ok=True)
    {'collect': collect, 'truth': truth, 'test': test, 'cmp': cmp, 'subset': subset}[a.what](a)
