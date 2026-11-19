"""In which kinds of position is our evaluation worse than Stockfish's network?

Reference: the deep Stockfish analysis of the quality test's positions (data/q/truth.jsonl, score of the best line from the
mover's side). Every evaluator's static value is brought to the reference's scale by its own best factor; the error is the
squared difference in expected score (1 / (1 + exp(-centipawns / 200))). Positions are sorted into overlapping categories
and the categories are listed by how much more error ours makes there than Stockfish 14.1's network.

  python3 ext/brain_err.py <our network .ema.pt> [more of ours ...]
"""
import json, math, os, sys
import numpy as np, torch, chess
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sfteach

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
KBT = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13] + [14, 14, 15, 15] * 3, device=dev)
pc = torch.arange(768, device=dev) // 64; sq = torch.arange(768, device=dev) % 64
LUT = torch.cat([((pc + 6) % 12) * 64 + (sq ^ 56), torch.tensor([768], device=dev)])
PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32, device=dev).repeat_interleave(64), torch.zeros(1, device=dev)])


def ours(path):
    w = {k: v.to(dev).float() for k, v in torch.load(path, map_location='cpu').items()}
    H = w['b'].shape[0]; NKB = (w['E'].shape[0] - 1) // 768; NEX = w['wo'].shape[0]
    Ef = torch.cat([w['E'][:NKB * 768] + w['Esh'][:768].repeat(NKB, 1), torch.zeros(1, H, device=dev)])

    def kfeat(idx):
        isk = (idx >= 320) & (idx < 384); ks = ((idx * isk).sum(1) - 320).clamp(0, 63)
        mir = ((ks & 7) >= 4).long() * 7; kq = ks ^ mir; kb = KBT[(kq >> 3) * 4 + (kq & 7)]
        return torch.where(idx == 768, torch.full_like(idx, NKB * 768), kb[:, None] * 768 + (idx // 64) * 64 + ((idx % 64) ^ mir[:, None]))

    @torch.no_grad()
    def f(idx):
        ao = (w['b'] + F.embedding_bag(kfeat(idx), Ef, mode='sum', padding_idx=NKB * 768)).clamp(0, 1) ** 2
        ap = (w['b'] + F.embedding_bag(kfeat(LUT[idx]), Ef, mode='sum', padding_idx=NKB * 768)).clamp(0, 1) ** 2
        bk = (((idx < 768).sum(1) - 2) // 4).clamp(0, NEX - 1)
        lin = w['c'] + F.embedding_bag(idx, w['S'], mode='sum', padding_idx=768); ph = (PT_W[idx].sum(1) / 24).clamp(max=1.0)
        return ph * lin[:, 0] + (1 - ph) * lin[:, 1] + (ao * w['wo'][bk]).sum(1) + (ap * w['wp'][bk]).sum(1) + w['cb'][bk]
    return f


VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def cats(b, t):
    """the categories a position belongs to"""
    us, them = b.turn, not b.turn; pm = b.piece_map(); n = len(pm); out = []
    out.append('pieces: ' + ('up to 10' if n <= 10 else '11-16' if n <= 16 else '17-22' if n <= 22 else '23-28' if n <= 28 else '29-32'))
    q = (len(b.pieces(chess.QUEEN, us)) > 0, len(b.pieces(chess.QUEEN, them)) > 0)
    out.append('queens: ' + ('both sides' if all(q) else 'none' if not any(q) else 'one side only'))
    mat = {c: sum(VAL[p.piece_type] for p in pm.values() if p.color == c) for c in (us, them)}; d = abs(mat[us] - mat[them])
    out.append('material difference: ' + ('none' if d == 0 else 'one pawn' if d == 1 else 'two or more pawns'))
    comp = lambda c: tuple(len(b.pieces(pt, c)) for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
    if comp(us) != comp(them): out.append('pieces of different kinds on the two sides')
    bs = [list(b.pieces(chess.BISHOP, c)) for c in (us, them)]
    if len(bs[0]) == 1 and len(bs[1]) == 1 and (chess.square_file(bs[0][0]) + chess.square_rank(bs[0][0])) % 2 != (chess.square_file(bs[1][0]) + chess.square_rank(bs[1][0])) % 2:
        out.append('opposite-coloured bishops')
    if b.is_check(): out.append('mover in check')
    # pieces en prise: attacked by something cheaper, or attacked and not defended
    def en_prise(c):
        k = 0
        for s, p in pm.items():
            if p.color != c or p.piece_type in (chess.PAWN, chess.KING): continue
            att = b.attackers(not c, s)
            if att and (min(VAL[b.piece_at(a).piece_type] or 99 for a in att) < VAL[p.piece_type] or not b.attackers(c, s)): k += 1
        return k
    eo, et = en_prise(us), en_prise(them)
    out.append("mover's pieces en prise: " + ('none' if eo == 0 else 'one' if eo == 1 else 'two or more'))
    out.append("opponent's pieces en prise: " + ('none' if et == 0 else 'one or more'))
    # pressure: pieces attacked at all (defended or not)
    ao = sum(1 for s, p in pm.items() if p.color == us and p.piece_type not in (chess.PAWN, chess.KING) and b.attackers(them, s))
    at = sum(1 for s, p in pm.items() if p.color == them and p.piece_type not in (chess.PAWN, chess.KING) and b.attackers(us, s))
    out.append('pieces under attack, both sides together: ' + ('0-1' if ao + at <= 1 else '2-3' if ao + at <= 3 else '4 or more'))
    # king exposure: enemy attacks on the squares around the king
    for c, nm in ((us, "mover's"), (them, "opponent's")):
        ks = b.king(c); ring = [s for s in chess.SQUARES if chess.square_distance(s, ks) == 1]; hit = sum(1 for s in ring if b.attackers(not c, s))
        out.append(f"{nm} king: squares around it attacked " + ('0' if hit == 0 else '1-2' if hit <= 2 else '3 or more'))
    # passed pawns far advanced
    def passed(c):
        for s in b.pieces(chess.PAWN, c):
            r = chess.square_rank(s) if c else 7 - chess.square_rank(s); f_ = chess.square_file(s)
            if r < 4: continue
            ahead = [x for x in b.pieces(chess.PAWN, not c) if abs(chess.square_file(x) - f_) <= 1 and ((chess.square_rank(x) > chess.square_rank(s)) if c else (chess.square_rank(x) < chess.square_rank(s)))]
            if not ahead: return True
        return False
    if passed(us) or passed(them): out.append('a passed pawn on the fifth rank or beyond')
    np_ = len(b.pieces(chess.PAWN, us)) + len(b.pieces(chess.PAWN, them))
    out.append('pawns: ' + ('up to 8' if np_ <= 8 else '9-12' if np_ <= 12 else '13-16'))
    mo = b.legal_moves.count(); b.push(chess.Move.null()); mt = b.legal_moves.count() if not b.is_check() else mo; b.pop()
    out.append('mobility: ' + ('mover has 10+ more moves' if mo - mt >= 10 else 'opponent has 10+ more moves' if mt - mo >= 10 else 'about even'))
    a_ = abs(t['cp1']); out.append('deep score: ' + ('within half a pawn' if a_ < 50 else 'half to one and a half pawns' if a_ < 150 else 'more than one and a half pawns'))
    m1 = chess.Move.from_uci(t['m1']); out.append('best move: ' + ('a capture or promotion' if b.is_capture(m1) or m1.promotion else 'a check' if b.gives_check(m1) else 'quiet'))
    if t['cp2'] is not None: out.append('second-best move: ' + ('within 0.04 of the best' if E(t['cp1']) - E(t['cp2']) < 0.04 else 'clearly worse (one good move)'))
    out.append('castling: ' + ('still possible for someone' if b.castling_rights else 'no rights left'))
    return out


E = lambda cp: 1 / (1 + math.exp(-max(-4000, min(4000, cp)) / 200))
tr = [t for t in map(json.loads, open('data/q/truth.jsonl')) if t['nodes'] > 50000 and abs(t['cp1']) <= 800]
idx = np.full((len(tr), 32), 768, np.int64); C = []
for i, t in enumerate(tr):
    b = chess.Board(t['fen']); k = 0
    for s, p in b.piece_map().items():
        mine = p.color == b.turn; ss = s if b.turn else s ^ 56; c = p.piece_type - 1
        idx[i, k] = (c if mine else c + 6) * 64 + ss; k += 1
    C.append(cats(b, t))
X = torch.from_numpy(idx).to(dev); P = np.array([E(t['cp1']) for t in tr])
evs = {os.path.basename(p).replace('.bin.ema.pt', '').replace('w_', ''): ours(p)(X).cpu().numpy() for p in sys.argv[1:]}
sf = sfteach.load('/root/ext/sf141_embedded.bin', dev)
with torch.no_grad():
    evs['Stockfish 14.1 network'] = sf(X)[1].cpu().numpy().astype(np.float64)
err = {}
print(f'{len(tr)} positions with a deep analysis')
for name, z in evs.items():
    best = min(((float(np.mean((1 / (1 + np.exp(-a * z)) - P) ** 2)), a) for a in np.exp(np.linspace(math.log(1e-4), math.log(30), 400))))
    err[name] = (1 / (1 + np.exp(-best[1] * z)) - P); print(f'{name:24s} best factor {best[1]:.4g}: mean squared error {1000 * best[0]:.3f} thousandths; mean error {1000 * float(err[name].mean()):+.2f} (positive = too hopeful for the mover)')
names = list(evs); me, ref = names[0], 'Stockfish 14.1 network'
rows = {}
for i, cs in enumerate(C):
    for c in cs: rows.setdefault(c, []).append(i)
print(f'\n{"category":58s} {"positions":>9s}  ' + '  '.join(f'{n[:12]:>12s}' for n in names) + f'   {me} minus Stockfish, per 1000 positions of the whole set')
tab = []
for c, ii in rows.items():
    ii = np.array(ii); ms = [1000 * float(np.mean(err[n][ii] ** 2)) for n in names]
    tab.append((1000 * float(np.sum(err[me][ii] ** 2 - err[ref][ii] ** 2)) / len(tr) * 1000 / 1000, c, len(ii), ms))
for ex, c, n, ms in sorted(tab, key=lambda r: r[1]):
    print(f'{c:58s} {n:9d}  ' + '  '.join(f'{m:12.3f}' for m in ms) + f'   {ex:+8.3f}   bias {1000 * float(np.mean(err[me][np.array(rows[c])])):+6.1f} / {1000 * float(np.mean(err[ref][np.array(rows[c])])):+6.1f}')
