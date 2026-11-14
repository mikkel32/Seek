"""Canonical board / move encoding shared by data prep, training, and the engine.

Canonical frame: the side to move is always "own", moving up the board.
  squares: a1=0 ... h8=63 after (optional) vertical flip
  pieces : 0 empty, 1..6 own P N B R Q K, 7..12 opp P N B R Q K
  castle : bit0 own K-side, bit1 own Q-side, bit2 opp K-side, bit3 opp Q-side
  ep     : canonical square index of the en-passant target, 64 = none
Move index: from*64+to (queen promotion implied), under-promotions
  4096 + (from_file*8 + to_file)*3 + {n:0,b:1,r:2}
"""
import math

N_MOVES = 4096 + 192

_EXPAND = str.maketrans({str(i): '.' * i for i in range(1, 9)})
_W = {'.': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
      'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}
_TW = str.maketrans({c: chr(v) for c, v in _W.items()})
_TB = str.maketrans({c: chr(v if v == 0 else (v + 6 if v <= 6 else v - 6)) for c, v in _W.items()})


def encode_fen(fen):
    """-> (board: 64 bytes, castle: int, ep: int, black: bool)"""
    parts = fen.split(' ')
    rows = parts[0].translate(_EXPAND).split('/')
    black = parts[1] == 'b'
    if black:
        board = ''.join(rows).translate(_TB).encode('latin1')
    else:
        board = ''.join(rows[::-1]).translate(_TW).encode('latin1')
    c = parts[2]
    if black:
        castle = ('k' in c) | (('q' in c) << 1) | (('K' in c) << 2) | (('Q' in c) << 3)
    else:
        castle = ('K' in c) | (('Q' in c) << 1) | (('k' in c) << 2) | (('q' in c) << 3)
    ep = parts[3] if len(parts) > 3 else '-'
    if ep == '-':
        e = 64
    else:
        e = (ord(ep[0]) - 97) + 8 * (ord(ep[1]) - 49)
        if black:
            e ^= 56
    return board, castle, e, black


def encode_move(uci, black, board):
    f = (ord(uci[0]) - 97) + 8 * (ord(uci[1]) - 49)
    t = (ord(uci[2]) - 97) + 8 * (ord(uci[3]) - 49)
    if black:
        f ^= 56
        t ^= 56
    # castling written as king-takes-rook -> standard king destination
    if f == 4 and board[f] == 6 and board[t] == 4 and (t == 7 or t == 0):
        t = 6 if t == 7 else 2
    if len(uci) == 5 and uci[4] != 'q':
        return 4096 + ((f & 7) * 8 + (t & 7)) * 3 + 'nbr'.index(uci[4])
    return f * 64 + t


def decode_move(idx, black, board=None):
    """canonical move index -> uci string in the real frame (board: canonical bytes, for queen promos)"""
    promo = ''
    if idx >= 4096:
        r = idx - 4096
        p = r % 3
        ft = r // 3
        f = 48 + ft // 8
        t = 56 + ft % 8
        promo = 'nbr'[p]
    else:
        f, t = idx // 64, idx % 64
        if board is not None and board[f] == 1 and t >= 56:
            promo = 'q'
    if black:
        f ^= 56
        t ^= 56
    return 'abcdefgh'[f & 7] + str((f >> 3) + 1) + 'abcdefgh'[t & 7] + str((t >> 3) + 1) + promo


def cp_to_wp(cp):
    return 1.0 / (1.0 + math.exp(-0.00368208 * cp))
