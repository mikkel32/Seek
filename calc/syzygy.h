/* Optional Syzygy endgame support for the single-threaded CALC search. */
#include <tbprobe.h>

static unsigned tb_hits;

static inline int tb_eligible(void) {
    return TB_LARGEST && !P.castle && (unsigned)popc(P.occ[WHITE] | P.occ[BLACK]) <= TB_LARGEST;
}

// Fathom and CALC both number a1 as square zero. Fathom uses zero for no EP square.
#define SEEK_TB_BOARD P.occ[WHITE], P.occ[BLACK], (P.bb[WHITE][KING] | P.bb[BLACK][KING]), \
    (P.bb[WHITE][QUEEN] | P.bb[BLACK][QUEEN]), (P.bb[WHITE][ROOK] | P.bb[BLACK][ROOK]), \
    (P.bb[WHITE][BISHOP] | P.bb[BLACK][BISHOP]), (P.bb[WHITE][KNIGHT] | P.bb[BLACK][KNIGHT]), \
    (P.bb[WHITE][PAWN] | P.bb[BLACK][PAWN]), P.half, P.castle, (P.ep == NOSQ ? 0u : (unsigned)P.ep), P.side == WHITE

// A WDL-only win/loss is a bound. Search on when the window can distinguish it
// from a shorter mate. A fresh 50-move clock makes WDL usable without DTZ.
static int tb_cutoff(int alpha, int beta, int ply, int *score) {
    if (P.half || !tb_eligible()) return 0;
    unsigned wdl = tb_probe_wdl(SEEK_TB_BOARD);
    if (wdl == TB_RESULT_FAILED) return 0;
    tb_hits++;
    const int win = MATE_IN - MAXPLY - ply;
    if (wdl == TB_WIN) { *score = win; return win >= beta; }
    if (wdl == TB_LOSS) { *score = -win; return -win <= alpha; }
    *score = 0;
    return 1;
}

static int tb_same_move(unsigned result, Move move) {
    static const int promotion[] = {0, QUEEN, ROOK, BISHOP, KNIGHT};
    unsigned p = TB_GET_PROMOTES(result);
    return p < sizeof(promotion) / sizeof(*promotion) && promotion[p] == PROMO(move) &&
           TB_GET_FROM(result) == (unsigned)FROM(move) && TB_GET_TO(result) == (unsigned)TO(move);
}

// Play a shortest route to a zeroing move in won positions. When drawn,
// keep every draw-preserving move and let the usual search choose among them.
static Move tb_root_moves(Move *moves, int *count, int *score) {
    if (!tb_eligible()) return 0;
    unsigned alternatives[TB_MAX_MOVES];
    unsigned result = tb_probe_root(SEEK_TB_BOARD, alternatives);
    if (result == TB_RESULT_FAILED || result == TB_RESULT_STALEMATE || result == TB_RESULT_CHECKMATE) return 0;
    unsigned wdl = TB_GET_WDL(result);
    tb_hits++;
    if (wdl == TB_LOSS) return 0;
    if (wdl != TB_DRAW) {
        for (int i = 0; i < *count; i++) if (tb_same_move(result, moves[i])) {
            *score = wdl == TB_WIN ? MATE_IN - MAXPLY : 0;
            moves[0] = moves[i]; *count = 1;
            return moves[0];
        }
        return 0;
    }
    int kept = 0;
    for (int i = 0; i < *count; i++) {
        for (int j = 0; alternatives[j] != TB_RESULT_FAILED; j++) {
            if (TB_GET_WDL(alternatives[j]) == TB_DRAW && tb_same_move(alternatives[j], moves[i])) {
                moves[kept++] = moves[i];
                break;
            }
        }
    }
    if (kept) *count = kept;
    return 0;
}
#undef SEEK_TB_BOARD
