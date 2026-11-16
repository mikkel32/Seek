"""Which kinds of best move does a configuration miss?  (data of ext/quality.py)

For the one-good-move positions (Stockfish's second line at least 0.04 worse than its first), the best move is put in a
class (capture that wins or trades / capture that gives up material at first sight / quiet check / quiet move onto an
attacked square / pawn push to the 6th or 7th rank / other quiet move) and each tested run's solving rate is printed per class.

  python3 ext/q_classes.py bigm_300000 sf_300000 bigm_1000000 ...
"""
import json, math, sys
import chess

D = 'data/q'; VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 99}
E = lambda cp: 1 / (1 + math.exp(-cp / 200))


def klass(b, m):
    cap = b.is_capture(m); chk = b.gives_check(m); pc = b.piece_at(m.from_square)
    if m.promotion: return 'promotion'
    if cap:
        vic = chess.PAWN if b.is_en_passant(m) else b.piece_at(m.to_square).piece_type
        if VAL[vic] >= VAL[pc.piece_type] or not b.is_attacked_by(not b.turn, m.to_square): return 'capture, safe' + (' +check' if chk else '')
        return 'capture, risky' + (' +check' if chk else '')
    att = b.attackers(not b.turn, m.to_square)
    cheaper = any(VAL[b.piece_at(s).piece_type] < VAL[pc.piece_type] for s in att)
    b.push(m); defended = b.is_attacked_by(not b.turn, m.to_square); b.pop()
    offered = pc.piece_type != chess.KING and att and (cheaper or not defended)
    if chk: return 'quiet check' + (', piece offered' if offered else '')
    if offered: return 'quiet, piece offered'
    if pc.piece_type == chess.PAWN and chess.square_rank(m.to_square) in ((5, 6) if b.turn else (2, 1)): return 'pawn to 6th/7th'
    if pc.piece_type == chess.KING: return 'king move'
    return 'quiet, other'


truth = {}
for l in open(f'{D}/truth.jsonl'):
    t = json.loads(l)
    if abs(t['cp1']) <= 500 and t['nodes'] > 50000 and t['cp2'] is not None and E(t['cp1']) - E(t['cp2']) >= 0.04: truth[t['fen']] = t
runs = {r: json.load(open(f'{D}/runs/{r}.json')) for r in sys.argv[1:]}
moves = {r: dict(zip(x['fens'], x['moves'])) for r, x in runs.items()}
fens = [f for f in truth if all(f in moves[r] for r in runs)]
cls = {}
for f in fens:
    b = chess.Board(f); cls.setdefault(klass(b, chess.Move.from_uci(truth[f]['m1'])), []).append(f)
print(f'{len(fens)} one-good-move positions; share solved per class of the best move')
print(f'{"class":32s} {"n":>5s} ' + ' '.join(f'{r[-14:]:>14s}' for r in runs))
for k, fs in sorted(cls.items(), key=lambda kv: -len(kv[1])):
    print(f'{k:32s} {len(fs):5d} ' + ' '.join(f'{100 * sum(moves[r][f] == truth[f]["m1"] for f in fs) / len(fs):13.1f}%' for r in runs))
print(f'{"all":32s} {len(fens):5d} ' + ' '.join(f'{100 * sum(moves[r][f] == truth[f]["m1"] for f in fens) / len(fens):13.1f}%' for r in runs))
