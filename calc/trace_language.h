// Causal sequence of one token tuple per legal root move per completed search depth.
// Candidate slots distinguish underpromotions. Training and deployment use this encoder.
#define TR_FIELDS 7
#define TR_TAPS 3
#define TR_VOCAB 68
enum { TR_LEAD=0, TR_GAP=3, TR_COST=13, TR_RANK=28, TR_ROLE=39, TR_THREAT=53, TR_TREND=58 };
static unsigned char tr_hist[256][TR_TAPS][TR_FIELDS], tr_kind[256], tr_threat[256];
static unsigned char tr_slots[5*4096];
static Move tr_moves[256];
#ifdef SEEK_TRACE_EXTEND
static Move tr_verify_move;
#endif
static U64 tr_work[256];
static inline int tr_key(Move m) { return 4096 * PROMO(m) + 64 * FROM(m) + TO(m); }
static inline int tr_slot(Move m) { return tr_slots[tr_key(m)]; }
static void tr_start(void) {
    memset(tr_hist, 0, sizeof tr_hist);
#ifdef SEEK_TRACE_EXTEND
    tr_verify_move = 0;
#endif
    U64 danger = attacks_by(P.side ^ 1);
    for (int i = 0; i < n_root; i++) {
        Move m = root_moves[i];
        tr_slots[tr_key(m)] = (unsigned char)i;
        tr_moves[i] = m;
        tr_kind[i] = PROMO(m) ? 12 : (P.sq[FROM(m)] % 6 + 6 * !!is_capture(m));
        tr_threat[i] = !!(danger & BIT(FROM(m))) + 2 * !!(danger & BIT(TO(m)));
    }
#ifdef SEEK_TRACE_COLLECT
    printf("trstart %d", n_root);
    for (int i = 0; i < n_root; i++) { char ms[8]; move_str(tr_moves[i], ms); printf(" %s", ms); }
    putchar('\n');
#endif
}
static int tr_gap_bucket(int gap) {
    static const int cut[] = {0,2,6,15,35,75,150,320};
    int i = 0; while (i < 8 && gap > cut[i]) i++;
    return i;
}
static void tr_complete(Move leader, int value, int previous, int depth) {
    U64 total = 1;
    for (int i = 0; i < n_root; i++) total += tr_work[i];
    int delta = value - previous;
    int trend = depth == 1 ? 0 : 1 + (delta > -80) + (delta > -30) + (delta > -8) +
                                      (delta > 8) + (delta > 30) + (delta > 80);
    for (int i = 0; i < n_root; i++) {
        int s = tr_slot(root_moves[i]);
        unsigned char (*history)[TR_FIELDS] = tr_hist[s];
        memmove(history[1], history[0], (TR_TAPS - 1) * TR_FIELDS);
        unsigned char *t = history[0];
        int relative = tr_work[s] ? msb(total) - msb(tr_work[s]) : 13;
        t[0] = TR_LEAD + 1 + (root_moves[i] == leader);
        t[1] = TR_GAP + tr_gap_bucket(value - root_scores[i]);
        t[2] = TR_COST + (relative > 13 ? 13 : relative);
        t[3] = TR_RANK + (i < 9 ? i : 9);
        t[4] = TR_ROLE + tr_kind[s];
        t[5] = TR_THREAT + tr_threat[s];
        t[6] = TR_TREND + trend;
    }
#ifdef SEEK_TRACE_COLLECT
    static const char hex[] = "0123456789abcdef";
    printf("trstep %d %d ", depth, tr_slot(leader));
    for (int s = 0; s < n_root; s++) for (int f = 0; f < TR_FIELDS; f++) {
        unsigned char v = tr_hist[s][0][f]; putchar(hex[v >> 4]); putchar(hex[v & 15]);
    }
    putchar('\n');
#endif
}
#ifdef SEEK_TRACE_MODEL
#include "trace_weights.h"
static int tr_logits[256], tr_lead_logits[256];
#ifdef SEEK_TRACE_CLOCK
static double tr_clock_scale = 1.0;
#endif
static void tr_predict(void) {
    for (int s = 0; s < n_root; s++) {
        int acc[TR_HIDDEN] = {0};
        for (int k = 0; k < TR_TAPS; k++) for (int f = 0; f < TR_FIELDS; f++) {
            const short *v = tr_embed[k][tr_hist[s][k][f]];
            for (int h = 0; h < TR_HIDDEN; h++) acc[h] += v[h];
        }
        int challenger = 0, next = 0;
        for (int h = 0; h < TR_HIDDEN; h++) {
            int z = acc[h] + tr_bias[h];
            z = z < 0 ? 0 : (z > TR_CLIP ? TR_CLIP : z);
            challenger += z * tr_challenger[h];
            next += z * tr_next[h];
        }
        tr_logits[s] = challenger; tr_lead_logits[s] = next;
    }
}
// Keep the proven leader first. The model nominates challengers for ordinary PVS.
static void tr_order(int depth) {
    if (depth < 2 || n_root < 3) return;
    tr_predict();
#ifdef SEEK_TRACE_COLLECT
    printf("trlog %d", depth);
    for (int s = 0; s < n_root; s++) printf(" %d,%d", tr_lead_logits[s], tr_logits[s]);
    putchar('\n');
#endif
#ifdef SEEK_TRACE_CLOCK
    if (tm_dynamic && time_ms_limit) {
        int leader_score = tr_lead_logits[tr_slot(root_moves[0])], maximum = leader_score;
        for (int i = 1; i < n_root; i++)
            if (tr_lead_logits[tr_slot(root_moves[i])] > maximum) maximum = tr_lead_logits[tr_slot(root_moves[i])];
        double mass = 0;
        for (int i = 0; i < n_root; i++) mass += exp((tr_lead_logits[tr_slot(root_moves[i])] - maximum)/(double)TR_LOGIT_SCALE);
        double hold = exp((leader_score - maximum)/(double)TR_LOGIT_SCALE) / mass;
        tr_clock_scale = fmin(1.65, .78 + 1.4 * (1.0 - hold));
    }
#endif
    int limit = SEEK_TRACE_MODEL < n_root - 1 ? SEEK_TRACE_MODEL : n_root - 1;
    for (int i = 1; i <= limit; i++) {
        int best = i;
        for (int j = i + 1; j < n_root; j++)
            if (tr_logits[tr_slot(root_moves[j])] > tr_logits[tr_slot(root_moves[best])]) best = j;
        Move m = root_moves[best];
        memmove(root_moves + i + 1, root_moves + i, (best-i)*sizeof(Move));
        root_moves[i] = m;
    }
#ifdef SEEK_TRACE_EXTEND
    tr_verify_move = 0;
    if (depth >= 3 && n_root >= 3) {
        int chosen = tr_slot(root_moves[1]), rival = tr_slot(root_moves[2]);
        for (int i = 3; i < n_root; i++)
            if (tr_logits[tr_slot(root_moves[i])] > tr_logits[rival]) rival = tr_slot(root_moves[i]);
        if (tr_logits[chosen] - tr_logits[rival] > TR_LOGIT_SCALE / 2 &&
            tr_lead_logits[tr_slot(root_moves[0])] - tr_lead_logits[chosen] < 3 * TR_LOGIT_SCALE)
            tr_verify_move = root_moves[1];
    }
#endif
#ifdef SEEK_TRACE_CHALLENGE
    // Only give up the first search slot when the forecast odds favor a particular move.
    int first = 0;
    for (int i = 1; i < n_root; i++)
        if (tr_lead_logits[tr_slot(root_moves[i])] > tr_lead_logits[tr_slot(root_moves[first])]) first = i;
    if (first && depth >= 4 && tr_lead_logits[tr_slot(root_moves[first])] -
                               tr_lead_logits[tr_slot(root_moves[0])] > TR_LOGIT_SCALE / 3) {
        Move mv = root_moves[first];
        memmove(root_moves+1, root_moves, first*sizeof(Move));
        root_moves[0] = mv;
    }
#endif
}
#endif
