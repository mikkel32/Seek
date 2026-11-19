// A positive result requires a legal checking move and a proof for every legal reply.
// Failed and interrupted probes are never entered into the main transposition table.
typedef struct {
    U64 key;
    unsigned int generation;
    int horizon;
} MateFailure;

static MateFailure mate_failures[32768];
static unsigned int mate_generation;
static U64 mate_limit;
static int mate_max_plies = 11, mate_share = 24, mate_spent;
static int mate_interrupted;
static Move mate_first;
static int mate_record, mate_length;
static Move mate_certificate[32768];

static void mate_append(Move m) {
    if (!mate_record) return;
    if (mate_length == 32768) { mate_interrupted = 1; return; }
    mate_certificate[mate_length++] = m;
}

static int mate_enter(void) {
    if (nodes >= mate_limit || (hard_ms_limit && elapsed_ms() >= hard_ms_limit)) {
        mate_interrupted = 1;
        return 0;
    }
    nodes++;
    mate_spent++;
    return 1;
}

static int mate_attack(int plies, int root);

// Returns zero if any defense escapes the proof; otherwise returns the maximum
// number of plies to mate plus one, so that immediate checkmate returns one.
static int mate_defend(int plies) {
    if (!mate_enter()) return 0;
    Move moves[256];
    int n = gen(moves, 0), bound = 1;
    int draw = is_draw();
    for (int i = 0; i < n; i++) {
        if (!make(moves[i])) continue;
        // Checkmate takes precedence over a possible draw claim at the parent.
        mate_append(moves[i]);
        int child = !draw && plies > 0 ? mate_attack(plies - 1, 0) : 0;
        unmake();
        if (!child) return 0;
        if (child + 1 > bound) bound = child + 1;
    }
    mate_append(0);  // end of the complete defense list, including a checkmated leaf
    return bound;
}

static int mate_attack(int plies, int root) {
    if (plies < 1 || !mate_enter() || is_draw()) return 0;
    MateFailure *entry = &mate_failures[P.key & 32767];
    if (entry->generation == mate_generation && entry->key == P.key && entry->horizon >= plies)
        return 0;
    Move moves[256];
    int n = gen(moves, 0);
    U64 targets[6], blockers;
    check_info(targets, &blockers);
    for (int i = 0; i < n; i++) {
        Move m = moves[i];
        int piece = P.sq[FROM(m)] % 6;
        if (!(targets[piece] & BIT(TO(m))) && !(blockers & BIT(FROM(m))) &&
            !PROMO(m) && FLAG(m) != F_EP && FLAG(m) != F_CASTLE) continue;
        if (!make(m)) continue;
        int mark = mate_length;
        mate_append(m);
        int child = in_check() ? mate_defend(plies - 1) : 0;
        unmake();
        if (mate_interrupted) return 0;
        if (child) {
            if (root) mate_first = m;
            return child + 1;
        }
        mate_length = mark;
    }
    // A cached failure can only suppress a proof. Successes depend on the
    // current repetition history and are always re-established on the board.
    entry->key = P.key;
    entry->generation = mate_generation;
    entry->horizon = plies;
    return 0;
}

static Move mate_probe(int max_depth, int *mate_plies, U64 allowance) {
    mate_first = 0;
    mate_length = 0;
    mate_spent = 0;
    mate_interrupted = 0;
    if (!allowance || max_depth < 1) return 0;
    if (++mate_generation == 0) {
        memset(mate_failures, 0, sizeof mate_failures);
        mate_generation = 1;
    }
    mate_limit = nodes + allowance;
    if (node_limit && mate_limit > node_limit) mate_limit = node_limit;
    for (int plies = 1; plies <= max_depth && !mate_interrupted; plies += 2) {
        int bound = mate_attack(plies, 1);
        if (bound) {
            *mate_plies = bound - 1;
            return mate_first;
        }
    }
    return 0;
}
