// Return a capture search score and the line to the evaluation that supplied it.
// Standing pat is a choice outside check. In check, try all legal evasions.
#define LEAF_PLY 16
static Move leaf_line[LEAF_PLY + 1][LEAF_PLY + 1];
static int leaf_length[LEAF_PLY + 1];

static int settle(int ply, int alpha, int beta) {
    leaf_length[ply] = 0;
    if (ply == LEAF_PLY) return evaluate();
    if (is_draw()) return 0;
    int check = in_check();
    int best = check ? -INF : evaluate();
    if (best >= beta) return best;
    if (best > alpha) alpha = best;
    Picker picker;
    mp_init(&picker, 0, ply, !check);
    Move move;
    while ((move = mp_next(&picker))) {
        if (!make(move)) continue;
        int score = -settle(ply + 1, -beta, -alpha);
        unmake();
        if (score <= best) continue;
        best = score;
        leaf_line[ply][0] = move;
        memcpy(leaf_line[ply] + 1, leaf_line[ply + 1], leaf_length[ply + 1] * sizeof(Move));
        leaf_length[ply] = 1 + leaf_length[ply + 1];
        if (best >= beta) break;
        if (best > alpha) alpha = best;
    }
    return best == -INF ? -MATE + ply : best;
}

