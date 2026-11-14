// CALC — the fast "calculator" half of the two-system engine.
// Bitboard move generation, PVS alpha-beta with a transposition table, null move, late-move reductions,
// quiescence with static exchange evaluation, and a small NNUE-style evaluation read from a weights file.
// Speaks a subset of UCI. Custom commands: perft N, eval, bonus <move> <cp> ... (root prior from OURO).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef unsigned long long U64;
typedef int Move;
enum { WHITE, BLACK };
enum { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PT };
enum { EMPTY = 12, NOSQ = 64 };
enum { F_NORMAL, F_EP, F_CASTLE, F_DPUSH };
#define FROM(m) ((m) & 63)
#define TO(m) (((m) >> 6) & 63)
#define PROMO(m) (((m) >> 12) & 7)
#define FLAG(m) (((m) >> 15) & 3)
#define MK(f, t, p, fl) ((f) | ((t) << 6) | ((p) << 12) | ((fl) << 15))
#define BIT(s) (1ULL << (s))
#define MAXPLY 100
#define INF 32000
#define MATE 31000
#define MATE_IN 30000

static inline int lsb(U64 b) { return __builtin_ctzll(b); }
static inline int msb(U64 b) { return 63 - __builtin_clzll(b); }
static inline int popc(U64 b) { return __builtin_popcountll(b); }

// ------------------------------------------------------------------ attack tables
enum { D_N, D_E, D_NE, D_NW, D_S, D_W, D_SE, D_SW };
static U64 rays[8][64], knight_att[64], king_att[64], pawn_att[2][64];
static int cmask[64];

static void init_tables(void) {
    static const int dr[8] = {1, 0, 1, 1, -1, 0, -1, -1}, df[8] = {0, 1, 1, -1, 0, -1, 1, -1};
    for (int s = 0; s < 64; s++) {
        int r = s >> 3, f = s & 7;
        for (int d = 0; d < 8; d++) {
            U64 b = 0;
            for (int rr = r + dr[d], ff = f + df[d]; rr >= 0 && rr < 8 && ff >= 0 && ff < 8; rr += dr[d], ff += df[d]) b |= BIT(rr * 8 + ff);
            rays[d][s] = b;
        }
        U64 k = 0, n = 0;
        for (int a = -2; a <= 2; a++) for (int b = -2; b <= 2; b++) {
            int rr = r + a, ff = f + b;
            if (rr < 0 || rr > 7 || ff < 0 || ff > 7) continue;
            if (abs(a) * abs(b) == 2) n |= BIT(rr * 8 + ff);
            if ((a || b) && abs(a) <= 1 && abs(b) <= 1) k |= BIT(rr * 8 + ff);
        }
        knight_att[s] = n; king_att[s] = k;
        pawn_att[WHITE][s] = pawn_att[BLACK][s] = 0;
        if (r < 7) { if (f > 0) pawn_att[WHITE][s] |= BIT(s + 7); if (f < 7) pawn_att[WHITE][s] |= BIT(s + 9); }
        if (r > 0) { if (f > 0) pawn_att[BLACK][s] |= BIT(s - 9); if (f < 7) pawn_att[BLACK][s] |= BIT(s - 7); }
        cmask[s] = 15;
    }
    cmask[4] = 15 & ~3; cmask[7] = 15 & ~1; cmask[0] = 15 & ~2;
    cmask[60] = 15 & ~12; cmask[63] = 15 & ~4; cmask[56] = 15 & ~8;
}
static inline U64 ray_pos(int d, int s, U64 occ) { U64 a = rays[d][s], b = a & occ; if (b) a ^= rays[d][lsb(b)]; return a; }
static inline U64 ray_neg(int d, int s, U64 occ) { U64 a = rays[d][s], b = a & occ; if (b) a ^= rays[d][msb(b)]; return a; }
static inline U64 bishop_att(int s, U64 occ) { return ray_pos(D_NE, s, occ) | ray_pos(D_NW, s, occ) | ray_neg(D_SE, s, occ) | ray_neg(D_SW, s, occ); }
static inline U64 rook_att(int s, U64 occ) { return ray_pos(D_N, s, occ) | ray_pos(D_E, s, occ) | ray_neg(D_S, s, occ) | ray_neg(D_W, s, occ); }

// ------------------------------------------------------------------ position
typedef struct { U64 bb[2][6], occ[2]; signed char sq[64]; int side, castle, ep, half; U64 key; } Pos;
typedef struct { Move m; int cap, castle, ep, half; U64 key; } Undo;
static Pos P;
static Undo ust[2048];
static U64 keys[2048];
static int hply;
static U64 zpiece[12][64], zcastle[16], zep[8], zside;

static U64 rng_state = 0x9E3779B97F4A7C15ULL;
static U64 rnd64(void) { U64 z = (rng_state += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
static void init_zobrist(void) {
    for (int p = 0; p < 12; p++) for (int s = 0; s < 64; s++) zpiece[p][s] = rnd64();
    for (int i = 0; i < 16; i++) zcastle[i] = rnd64();
    for (int i = 0; i < 8; i++) zep[i] = rnd64();
    zside = rnd64();
}
static inline void put(int c, int pt, int s) { P.bb[c][pt] |= BIT(s); P.occ[c] |= BIT(s); P.sq[s] = c * 6 + pt; P.key ^= zpiece[c * 6 + pt][s]; }
static inline void rem(int c, int pt, int s) { P.bb[c][pt] ^= BIT(s); P.occ[c] ^= BIT(s); P.sq[s] = EMPTY; P.key ^= zpiece[c * 6 + pt][s]; }

static int attacked(int s, int by) {
    if (pawn_att[by ^ 1][s] & P.bb[by][PAWN]) return 1;
    if (knight_att[s] & P.bb[by][KNIGHT]) return 1;
    if (king_att[s] & P.bb[by][KING]) return 1;
    U64 occ = P.occ[0] | P.occ[1];
    if (bishop_att(s, occ) & (P.bb[by][BISHOP] | P.bb[by][QUEEN])) return 1;
    if (rook_att(s, occ) & (P.bb[by][ROOK] | P.bb[by][QUEEN])) return 1;
    return 0;
}
static inline int in_check(void) { return attacked(lsb(P.bb[P.side][KING]), P.side ^ 1); }

static void unmake(void) {
    Undo *u = &ust[--hply];
    Move m = u->m; int f = FROM(m), t = TO(m), fl = FLAG(m);
    int them = P.side, us = them ^ 1;
    int pt = P.sq[t] % 6;
    rem(us, pt, t);
    put(us, PROMO(m) ? PAWN : pt, f);
    if (fl == F_EP) put(them, PAWN, t + (us == WHITE ? -8 : 8));
    else if (u->cap != EMPTY) put(them, u->cap % 6, t);
    if (fl == F_CASTLE) { if (t > f) { rem(us, ROOK, f + 1); put(us, ROOK, f + 3); } else { rem(us, ROOK, f - 1); put(us, ROOK, f - 4); } }
    P.side = us; P.castle = u->castle; P.ep = u->ep; P.half = u->half; P.key = u->key;
}
static int make(Move m) {
    int f = FROM(m), t = TO(m), fl = FLAG(m), pr = PROMO(m);
    int us = P.side, them = us ^ 1, pt = P.sq[f] % 6, cap = P.sq[t];
    Undo *u = &ust[hply]; u->m = m; u->cap = cap; u->castle = P.castle; u->ep = P.ep; u->half = P.half; u->key = P.key;
    keys[hply++] = P.key;
    if (fl == F_EP) rem(them, PAWN, t + (us == WHITE ? -8 : 8));
    else if (cap != EMPTY) rem(them, cap % 6, t);
    rem(us, pt, f); put(us, pr ? pr : pt, t);
    if (fl == F_CASTLE) { if (t > f) { rem(us, ROOK, f + 3); put(us, ROOK, f + 1); } else { rem(us, ROOK, f - 4); put(us, ROOK, f - 1); } }
    if (P.ep != NOSQ) P.key ^= zep[P.ep & 7];
    P.ep = NOSQ;
    if (fl == F_DPUSH && (pawn_att[us][(f + t) / 2] & P.bb[them][PAWN])) { P.ep = (f + t) / 2; P.key ^= zep[P.ep & 7]; }
    P.key ^= zcastle[P.castle]; P.castle &= cmask[f] & cmask[t]; P.key ^= zcastle[P.castle];
    P.half = (pt == PAWN || cap != EMPTY || fl == F_EP) ? 0 : P.half + 1;
    P.side = them; P.key ^= zside;
    if (attacked(lsb(P.bb[us][KING]), them)) { unmake(); return 0; }
    return 1;
}
static void make_null(void) {
    Undo *u = &ust[hply]; u->m = 0; u->cap = EMPTY; u->castle = P.castle; u->ep = P.ep; u->half = P.half; u->key = P.key;
    keys[hply++] = P.key;
    if (P.ep != NOSQ) P.key ^= zep[P.ep & 7];
    P.ep = NOSQ; P.half++; P.side ^= 1; P.key ^= zside;
}
static void unmake_null(void) { Undo *u = &ust[--hply]; P.side ^= 1; P.ep = u->ep; P.half = u->half; P.key = u->key; }

// ------------------------------------------------------------------ move generation (pseudo-legal)
static inline Move *add_promos(Move *l, int f, int t, int all) {
    *l++ = MK(f, t, QUEEN, 0);
    if (all) { *l++ = MK(f, t, KNIGHT, 0); *l++ = MK(f, t, ROOK, 0); *l++ = MK(f, t, BISHOP, 0); }
    return l;
}
static int gen(Move *list, int caps_only) {
    Move *l = list;
    int us = P.side, them = us ^ 1;
    U64 own = P.occ[us], opp = P.occ[them], occ = own | opp, empty = ~occ;
    U64 target = caps_only ? opp : ~own;
    U64 pawns = P.bb[us][PAWN];
    int up = us == WHITE ? 8 : -8;
    U64 r7 = us == WHITE ? 0x00FF000000000000ULL : 0x000000000000FF00ULL, r3 = us == WHITE ? 0x0000000000FF0000ULL : 0x0000FF0000000000ULL;
    U64 b = pawns;
    while (b) {
        int f = lsb(b); b &= b - 1;
        int promo = (BIT(f) & r7) != 0;
        U64 a = pawn_att[us][f] & opp;
        while (a) { int t = lsb(a); a &= a - 1; if (promo) l = add_promos(l, f, t, 1); else *l++ = MK(f, t, 0, 0); }
        if (P.ep != NOSQ && (pawn_att[us][f] & BIT(P.ep))) *l++ = MK(f, P.ep, 0, F_EP);
        int t = f + up;
        if (empty & BIT(t)) {
            if (promo) l = add_promos(l, f, t, !caps_only);
            else if (!caps_only) {
                *l++ = MK(f, t, 0, 0);
                if ((BIT(t) & r3) && (empty & BIT(t + up))) *l++ = MK(f, t + up, 0, F_DPUSH);
            }
        }
    }
    b = P.bb[us][KNIGHT];
    while (b) { int f = lsb(b); b &= b - 1; U64 a = knight_att[f] & target; while (a) { int t = lsb(a); a &= a - 1; *l++ = MK(f, t, 0, 0); } }
    b = P.bb[us][BISHOP] | P.bb[us][QUEEN];
    while (b) { int f = lsb(b); b &= b - 1; U64 a = bishop_att(f, occ) & target; while (a) { int t = lsb(a); a &= a - 1; *l++ = MK(f, t, 0, 0); } }
    b = P.bb[us][ROOK] | P.bb[us][QUEEN];
    while (b) { int f = lsb(b); b &= b - 1; U64 a = rook_att(f, occ) & target; while (a) { int t = lsb(a); a &= a - 1; *l++ = MK(f, t, 0, 0); } }
    int k = lsb(P.bb[us][KING]);
    U64 a = king_att[k] & target;
    while (a) { int t = lsb(a); a &= a - 1; *l++ = MK(k, t, 0, 0); }
    if (!caps_only) {
        int ks = us == WHITE ? 1 : 4, qs = us == WHITE ? 2 : 8, e = us == WHITE ? 4 : 60;
        if (k == e && (P.castle & (ks | qs)) && !attacked(e, them)) {
            if ((P.castle & ks) && !(occ & (BIT(e + 1) | BIT(e + 2))) && !attacked(e + 1, them)) *l++ = MK(e, e + 2, 0, F_CASTLE);
            if ((P.castle & qs) && !(occ & (BIT(e - 1) | BIT(e - 2) | BIT(e - 3))) && !attacked(e - 1, them)) *l++ = MK(e, e - 2, 0, F_CASTLE);
        }
    }
    return (int)(l - list);
}

// ------------------------------------------------------------------ evaluation (NNUE-style, from the mover's view)
#define HMAX 1024
static int NH = 0;                       // hidden width; 0 = linear piece-square evaluation only
static float *NE, NB[HMAX], NWO[HMAX], NWP[HMAX], NC = 0, NS[768], NC2 = 0, NS2[768];
static int have_net = 0, taper = 0;   // taper: second (endgame) linear table, blended by remaining material
static const int mat_val[7] = {100, 320, 330, 500, 950, 0, 0};

static int load_net(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int hdr[2];
    if (fread(hdr, 4, 2, f) != 2 || hdr[0] != 0x4F55524F || hdr[1] < 0 || hdr[1] > HMAX) { fclose(f); return 0; }
    NH = hdr[1];
    free(NE); NE = (float *)aligned_alloc(64, (size_t)(768 * (NH ? NH : 1) * 4 + 63) / 64 * 64);
    size_t ok = 1;
    if (NH) { ok &= fread(NE, 4, 768 * NH, f) == (size_t)(768 * NH); ok &= fread(NB, 4, NH, f) == (size_t)NH; ok &= fread(NWO, 4, NH, f) == (size_t)NH; ok &= fread(NWP, 4, NH, f) == (size_t)NH; }
    ok &= fread(&NC, 4, 1, f) == 1; ok &= fread(NS, 4, 768, f) == 768;
    taper = ok && fread(&NC2, 4, 1, f) == 1 && fread(NS2, 4, 768, f) == 768;
    fclose(f);
    have_net = (int)ok;
    return have_net;
}
static int evaluate(void) {
    int us = P.side;
    if (!have_net) {                     // fallback: bare material, so the engine runs without a weights file
        int v = 0;
        for (int pt = 0; pt < 5; pt++) v += mat_val[pt] * (popc(P.bb[us][pt]) - popc(P.bb[us ^ 1][pt]));
        return v;
    }
    float ao[HMAX] __attribute__((aligned(64))), ap[HMAX] __attribute__((aligned(64)));
    int n = NH;
    memcpy(ao, NB, n * 4); memcpy(ap, NB, n * 4);
    float lin = NC, lin2 = NC2;
    int flip = us == WHITE ? 0 : 56;
    for (int c = 0; c < 2; c++) {
        int rel = c == us ? 0 : 1;
        for (int pt = 0; pt < 6; pt++) {
            U64 b = P.bb[c][pt];
            int base_o = (rel * 6 + pt) * 64, base_p = ((1 - rel) * 6 + pt) * 64;
            while (b) {
                int s = lsb(b) ^ flip; b &= b - 1;
                int io = base_o + s, ip = base_p + (s ^ 56);
                lin += NS[io]; lin2 += NS2[io];
                const float *eo = NE + (size_t)io * n, *ep = NE + (size_t)ip * n;
                for (int i = 0; i < n; i++) ao[i] += eo[i];
                for (int i = 0; i < n; i++) ap[i] += ep[i];
            }
        }
    }
    if (taper) {
        int npm = popc(P.bb[0][KNIGHT] | P.bb[1][KNIGHT] | P.bb[0][BISHOP] | P.bb[1][BISHOP]) + 2 * popc(P.bb[0][ROOK] | P.bb[1][ROOK]) + 4 * popc(P.bb[0][QUEEN] | P.bb[1][QUEEN]);
        float ph = npm >= 24 ? 1.0f : npm / 24.0f;
        lin = ph * lin + (1.0f - ph) * lin2;
    }
    float out = lin;
    for (int i = 0; i < n; i++) { float x = ao[i] < 0 ? 0 : (ao[i] > 1 ? 1 : ao[i]); out += NWO[i] * x * x; }
    for (int i = 0; i < n; i++) { float x = ap[i] < 0 ? 0 : (ap[i] > 1 ? 1 : ap[i]); out += NWP[i] * x * x; }
    int v = (int)(out * 271.58f);        // logit of win probability -> centipawns
    return v > 20000 ? 20000 : (v < -20000 ? -20000 : v);
}

// ------------------------------------------------------------------ static exchange evaluation
static const int see_val[7] = {100, 320, 330, 500, 950, 20000, 0};
static int see(Move m) {
    int f = FROM(m), t = TO(m), gain[40], d = 0, pt = P.sq[f] % 6;
    int cap = FLAG(m) == F_EP ? PAWN : (P.sq[t] == EMPTY ? NO_PT : P.sq[t] % 6);
    gain[0] = see_val[cap];
    if (PROMO(m)) { gain[0] += see_val[PROMO(m)] - see_val[PAWN]; pt = PROMO(m); }
    U64 occ = (P.occ[0] | P.occ[1]) ^ BIT(f);
    if (FLAG(m) == F_EP) occ ^= BIT(t + (P.side == WHITE ? -8 : 8));
    U64 diag = P.bb[0][BISHOP] | P.bb[1][BISHOP] | P.bb[0][QUEEN] | P.bb[1][QUEEN];
    U64 orth = P.bb[0][ROOK] | P.bb[1][ROOK] | P.bb[0][QUEEN] | P.bb[1][QUEEN];
    U64 att = ((pawn_att[BLACK][t] & P.bb[WHITE][PAWN]) | (pawn_att[WHITE][t] & P.bb[BLACK][PAWN]) |
               (knight_att[t] & (P.bb[0][KNIGHT] | P.bb[1][KNIGHT])) | (king_att[t] & (P.bb[0][KING] | P.bb[1][KING])) |
               (bishop_att(t, occ) & diag) | (rook_att(t, occ) & orth)) & occ;
    int side = P.side ^ 1;
    for (;;) {
        U64 my = att & P.occ[side];
        if (!my) break;
        int apt; U64 b = 0;
        for (apt = PAWN; apt <= KING; apt++) { b = my & P.bb[side][apt]; if (b) break; }
        if (apt == KING && (att & P.occ[side ^ 1])) break;        // the king may not capture a defended piece
        d++; gain[d] = see_val[pt] - gain[d - 1];
        pt = apt; occ ^= b & -b;
        if (apt == PAWN || apt == BISHOP || apt == QUEEN) att |= bishop_att(t, occ) & diag;
        if (apt == ROOK || apt == QUEEN) att |= rook_att(t, occ) & orth;
        att &= occ; side ^= 1;
        if (d >= 38) break;
    }
    while (d > 0) { int a = -gain[d - 1], b2 = gain[d]; gain[d - 1] = -(a > b2 ? a : b2); d--; }
    return gain[0];
}

// ------------------------------------------------------------------ transposition table
typedef struct { U64 key; Move move; short score; unsigned char depth, flag; } TTE;
enum { TT_NONE, TT_EXACT, TT_LOWER, TT_UPPER };
static TTE *tt; static U64 tt_mask;
static void tt_alloc(int mb) { U64 n = 1; while (n * 2 * sizeof(TTE) <= (U64)mb << 20) n *= 2; free(tt); tt = (TTE *)calloc(n, sizeof(TTE)); tt_mask = n - 1; }

// ------------------------------------------------------------------ search
static U64 nodes, node_limit;
static int stop_flag, time_ms_limit, hard_ms_limit, root_depth;
static struct timespec t_start;
static Move killers[MAXPLY][2];
static int history_h[2][64][64];
static int lmr[64][64];
static int root_bonus[64 * 64];          // prior from the judgment network, centipawns, indexed by from*64+to

static int elapsed_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (int)((t.tv_sec - t_start.tv_sec) * 1000 + (t.tv_nsec - t_start.tv_nsec) / 1000000); }
static void check_time(void) { if ((hard_ms_limit && elapsed_ms() >= hard_ms_limit) || (node_limit && nodes >= node_limit)) stop_flag = 1; }

static int is_draw(void) {
    if (P.half >= 100) return 1;
    for (int i = hply - 2; i >= 0 && i >= hply - P.half; i -= 2) if (keys[i] == P.key) return 1;
    if (!(P.bb[0][PAWN] | P.bb[1][PAWN] | P.bb[0][ROOK] | P.bb[1][ROOK] | P.bb[0][QUEEN] | P.bb[1][QUEEN])) {
        if (popc(P.occ[0] | P.occ[1]) <= 3) return 1;                   // K v K, K+minor v K
    }
    return 0;
}
static inline int is_capture(Move m) { return P.sq[TO(m)] != EMPTY || FLAG(m) == F_EP; }

static void score_moves(Move *list, int *sc, int n, Move ttm, int ply) {
    for (int i = 0; i < n; i++) {
        Move m = list[i];
        if (m == ttm) sc[i] = 2000000000;
        else if (is_capture(m) || PROMO(m)) {
            int victim = FLAG(m) == F_EP ? PAWN : (P.sq[TO(m)] == EMPTY ? NO_PT : P.sq[TO(m)] % 6);
            int v = mat_val[victim] * 16 - mat_val[P.sq[FROM(m)] % 6] / 8 + (PROMO(m) == QUEEN ? 9000 : 0);
            sc[i] = (see(m) >= 0 ? 1000000000 : -1000000000) + v;
        } else if (m == killers[ply][0]) sc[i] = 900000000;
        else if (m == killers[ply][1]) sc[i] = 800000000;
        else sc[i] = history_h[P.side][FROM(m)][TO(m)];
    }
}
static inline Move pick(Move *list, int *sc, int n, int i) {
    int b = i;
    for (int j = i + 1; j < n; j++) if (sc[j] > sc[b]) b = j;
    Move m = list[b]; list[b] = list[i]; list[i] = m;
    int s = sc[b]; sc[b] = sc[i]; sc[i] = s;
    return m;
}

static int qsearch(int alpha, int beta, int ply) {
    if ((++nodes & 2047) == 0) check_time();
    if (stop_flag) return 0;
    if (ply >= MAXPLY - 1) return evaluate();
    int chk = in_check(), best, stand = -INF;
    if (!chk) {
        stand = best = evaluate();
        if (best >= beta) return best;
        if (best > alpha) alpha = best;
    } else best = -INF;
    Move list[256]; int sc[256];
    int n = gen(list, !chk), legal = 0;
    score_moves(list, sc, n, 0, ply);
    for (int i = 0; i < n; i++) {
        Move m = pick(list, sc, n, i);
        if (!chk) {
            if (sc[i] < 0) break;                                        // losing captures
            int victim = FLAG(m) == F_EP ? PAWN : (P.sq[TO(m)] == EMPTY ? NO_PT : P.sq[TO(m)] % 6);
            if (!PROMO(m) && stand + mat_val[victim] + 200 <= alpha) continue;   // delta pruning
        }
        if (!make(m)) continue;
        legal++;
        int v = -qsearch(-beta, -alpha, ply + 1);
        unmake();
        if (stop_flag) return 0;
        if (v > best) { best = v; if (v > alpha) { alpha = v; if (v >= beta) break; } }
    }
    if (chk && !legal) return -MATE + ply;
    return best;
}

static int search(int depth, int alpha, int beta, int ply, int allow_null) {
    int pv_node = beta - alpha > 1;
    if (is_draw()) return 0;
    int chk = in_check();
    if (chk) depth++;
    if (depth <= 0) return qsearch(alpha, beta, ply);
    if ((++nodes & 2047) == 0) check_time();
    if (stop_flag) return 0;
    if (ply >= MAXPLY - 1) return evaluate();
    if (alpha < -MATE + ply) alpha = -MATE + ply;                         // mate-distance pruning
    if (beta > MATE - ply - 1) beta = MATE - ply - 1;
    if (alpha >= beta) return alpha;

    TTE *e = &tt[P.key & tt_mask];
    Move ttm = 0;
    if (e->key == P.key) {
        ttm = e->move;
        if (!pv_node && e->depth >= depth) {
            int s = e->score;
            if (s > MATE_IN) s -= ply; else if (s < -MATE_IN) s += ply;
            if (e->flag == TT_EXACT || (e->flag == TT_LOWER && s >= beta) || (e->flag == TT_UPPER && s <= alpha)) return s;
        }
    }
    int ev = chk ? -INF : evaluate();
    if (!pv_node && !chk && beta > -MATE_IN && beta < MATE_IN) {
        if (depth <= 7 && ev - 75 * depth >= beta) return ev;             // reverse futility
        if (allow_null && depth >= 3 && ev >= beta &&
            (P.bb[P.side][KNIGHT] | P.bb[P.side][BISHOP] | P.bb[P.side][ROOK] | P.bb[P.side][QUEEN])) {
            int R = 3 + depth / 4 + ((ev - beta) / 200 > 2 ? 2 : (ev - beta) / 200);
            make_null();
            int v = -search(depth - 1 - R, -beta, -beta + 1, ply + 1, 0);
            unmake_null();
            if (stop_flag) return 0;
            if (v >= beta) return v > MATE_IN ? beta : v;
        }
    }
    Move list[256], quiets[256]; int sc[256];
    int n = gen(list, 0), legal = 0, nq = 0, best = -INF, a0 = alpha;
    Move bestm = 0;
    score_moves(list, sc, n, ttm, ply);
    for (int i = 0; i < n; i++) {
        Move m = pick(list, sc, n, i);
        int quiet = !is_capture(m) && !PROMO(m);
        if (!pv_node && !chk && quiet && best > -MATE_IN) {
            if (depth <= 4 && legal >= 3 + depth * depth) continue;       // late-move pruning
            if (depth <= 6 && ev + 80 + 80 * depth <= alpha) continue;    // futility
        }
        if (!pv_node && !chk && !quiet && depth <= 5 && sc[i] < 0 && best > -MATE_IN) continue;   // losing captures at low depth
        if (!make(m)) continue;
        legal++;
        int v, nd = depth - 1;
        if (legal == 1) v = -search(nd, -beta, -alpha, ply + 1, 1);
        else {
            int r = 0;
            if (depth >= 3 && quiet && legal > 1 + 2 * pv_node) {
                r = lmr[depth > 63 ? 63 : depth][legal > 63 ? 63 : legal];
                if (pv_node) r--;
                if (chk || in_check()) r--;
                if (m == killers[ply][0] || m == killers[ply][1]) r--;
                if (r < 0) r = 0;
                if (r > nd - 1) r = nd - 1;
            }
            v = -search(nd - r, -alpha - 1, -alpha, ply + 1, 1);
            if (v > alpha && r) v = -search(nd, -alpha - 1, -alpha, ply + 1, 1);
            if (v > alpha && v < beta) v = -search(nd, -beta, -alpha, ply + 1, 1);
        }
        unmake();
        if (stop_flag) return 0;
        if (v > best) {
            best = v; bestm = m;
            if (v > alpha) {
                alpha = v;
                if (v >= beta) {
                    if (quiet) {
                        if (killers[ply][0] != m) { killers[ply][1] = killers[ply][0]; killers[ply][0] = m; }
                        int bonus = depth * depth;
                        int *h = &history_h[P.side][FROM(m)][TO(m)];
                        *h += bonus - *h * bonus / 16384;
                        for (int q = 0; q < nq; q++) { int *g = &history_h[P.side][FROM(quiets[q])][TO(quiets[q])]; *g -= bonus + *g * bonus / 16384; }
                    }
                    break;
                }
            }
        }
        if (quiet && nq < 255) quiets[nq++] = m;
    }
    if (!legal) return chk ? -MATE + ply : 0;
    int s = best;
    if (s > MATE_IN) s += ply; else if (s < -MATE_IN) s -= ply;
    e->key = P.key; e->move = bestm; e->score = (short)s; e->depth = (unsigned char)depth;
    e->flag = best >= beta ? TT_LOWER : (best > a0 ? TT_EXACT : TT_UPPER);
    return best;
}

static void move_str(Move m, char *o) {
    int f = FROM(m), t = TO(m);
    o[0] = 'a' + (f & 7); o[1] = '1' + (f >> 3); o[2] = 'a' + (t & 7); o[3] = '1' + (t >> 3); o[4] = 0;
    if (PROMO(m)) { o[4] = "?nbrq"[PROMO(m)]; o[5] = 0; }
}

// root: iterative deepening, PVS over the root moves, optional additive prior (root_bonus) from the judgment network
static Move root_moves[256]; static int root_scores[256], n_root, use_bonus, print_root;
static Move search_root(int max_depth) {
    Move list[256];
    int n = gen(list, 0);
    n_root = 0;
    for (int i = 0; i < n; i++) if (make(list[i])) { unmake(); root_moves[n_root++] = list[i]; }
    if (!n_root) return 0;
    Move best = root_moves[0];
    memset(killers, 0, sizeof killers);
    for (int i = 0; i < 2; i++) for (int a = 0; a < 64; a++) for (int b = 0; b < 64; b++) history_h[i][a][b] /= 2;
    int last = 0;
    for (root_depth = 1; root_depth <= max_depth; root_depth++) {
        int alpha = -INF, beta = INF, iter_best = -INF; Move iter_move = 0;
        int delta = 25, first_done = 0;
        if (root_depth >= 5) { alpha = last - delta; beta = last + delta; }
        for (;;) {
            int a = alpha; iter_best = -INF; iter_move = 0; first_done = 0;
            for (int i = 0; i < n_root; i++) {
                Move m = root_moves[i];
                int bon = use_bonus ? root_bonus[FROM(m) * 64 + TO(m)] : 0;
                make(m);
                int v;
                if (i == 0) v = -search(root_depth - 1, -(beta - bon), -(a - bon), 1, 1) + bon;
                else {
                    v = -search(root_depth - 1, -(a - bon) - 1, -(a - bon), 1, 1) + bon;
                    if (v > a && !stop_flag) v = -search(root_depth - 1, -(beta - bon), -(a - bon), 1, 1) + bon;
                }
                unmake();
                if (stop_flag) break;
                root_scores[i] = v; first_done = 1;
                if (v > iter_best) { iter_best = v; iter_move = m; }
                if (v > a) { a = v; if (v >= beta) break; }
            }
            if (stop_flag) break;
            if (iter_best <= alpha) { alpha = iter_best - delta; if (alpha < -INF) alpha = -INF; delta *= 3; continue; }
            if (iter_best >= beta) { beta = iter_best + delta; if (beta > INF) beta = INF; delta *= 3;
                                     for (int i = 1; i < n_root; i++) if (root_moves[i] == iter_move) { memmove(root_moves + 1, root_moves, i * sizeof(Move)); root_moves[0] = iter_move; break; }
                                     continue; }
            break;
        }
        if (stop_flag) { if (first_done && iter_move && root_depth > 1) best = iter_move; break; }   // partial iteration: usable once the previous best has been re-searched
        best = iter_move; last = iter_best;
        for (int i = 1; i < n_root; i++) if (root_moves[i] == best) { memmove(root_moves + 1, root_moves, i * sizeof(Move)); root_moves[0] = best; break; }
        char ms[8]; move_str(best, ms);
        int t = elapsed_ms();
        if (last > MATE_IN) printf("info depth %d score mate %d nodes %llu time %d pv %s\n", root_depth, (MATE - last + 1) / 2, nodes, t, ms);
        else if (last < -MATE_IN) printf("info depth %d score mate %d nodes %llu time %d pv %s\n", root_depth, -(MATE + last + 1) / 2, nodes, t, ms);
        else printf("info depth %d score cp %d nodes %llu time %d pv %s\n", root_depth, last, nodes, t, ms);
        fflush(stdout);
        if (time_ms_limit && t >= time_ms_limit) break;
        if (n_root == 1 && root_depth >= 2) break;
        if (last > MATE_IN && root_depth >= 6) break;
    }
    return best;
}

// ------------------------------------------------------------------ setup / UCI
static void set_fen(const char *fen) {
    memset(&P, 0, sizeof P);
    for (int i = 0; i < 64; i++) P.sq[i] = EMPTY;
    int r = 7, f = 0; const char *c = fen;
    for (; *c && *c != ' '; c++) {
        if (*c == '/') { r--; f = 0; }
        else if (*c >= '1' && *c <= '8') f += *c - '0';
        else { const char *pcs = "PNBRQKpnbrqk"; const char *q = strchr(pcs, *c); if (q) { int i = (int)(q - pcs); put(i / 6, i % 6, r * 8 + f); } f++; }
    }
    while (*c == ' ') c++;
    P.side = *c == 'b' ? BLACK : WHITE; if (*c) c++;
    while (*c == ' ') c++;
    P.castle = 0;
    for (; *c && *c != ' '; c++) { if (*c == 'K') P.castle |= 1; if (*c == 'Q') P.castle |= 2; if (*c == 'k') P.castle |= 4; if (*c == 'q') P.castle |= 8; }
    while (*c == ' ') c++;
    P.ep = NOSQ;
    if (*c >= 'a' && *c <= 'h' && c[1]) { P.ep = (c[1] - '1') * 8 + (*c - 'a'); c += 2; } else if (*c) c++;
    while (*c == ' ') c++;
    P.half = atoi(c);
    // recompute the key from scratch
    P.key = 0;
    for (int s = 0; s < 64; s++) if (P.sq[s] != EMPTY) P.key ^= zpiece[(int)P.sq[s]][s];
    P.key ^= zcastle[P.castle];
    if (P.ep != NOSQ) {                                                    // keep ep only if a capture is possible (matches common hashing)
        if (pawn_att[P.side ^ 1][P.ep] & P.bb[P.side][PAWN]) P.key ^= zep[P.ep & 7]; else P.ep = NOSQ;
    }
    if (P.side == BLACK) P.key ^= zside;
    hply = 0;
}
static Move parse_move(const char *s) {
    Move list[256]; int n = gen(list, 0); char ms[8];
    for (int i = 0; i < n; i++) { move_str(list[i], ms); if (!strncmp(ms, s, strlen(ms)) && (s[strlen(ms)] == 0 || s[strlen(ms)] == ' ' || s[strlen(ms)] == '\n')) return list[i]; }
    return 0;
}
static U64 perft(int d) {
    if (d == 0) return 1;
    Move list[256]; int n = gen(list, 0); U64 t = 0;
    for (int i = 0; i < n; i++) if (make(list[i])) { t += perft(d - 1); unmake(); }
    return t;
}

int main(int argc, char **argv) {
    init_tables(); init_zobrist(); tt_alloc(32);
    for (int d = 1; d < 64; d++) for (int i = 1; i < 64; i++) lmr[d][i] = (int)(0.75 + log(d) * log(i) / 2.25);
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    if (argc > 1) load_net(argv[1]);
    static char line[65536];
    setvbuf(stdout, NULL, _IOLBF, 0);
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!strcmp(line, "uci")) {
            printf("id name CALC\nid author loopchess\noption name EvalFile type string default <none>\noption name Hash type spin default 32 min 1 max 4096\noption name Bonus type string default <none>\nuciok\n");
        } else if (!strcmp(line, "isready")) printf("readyok\n");
        else if (!strncmp(line, "setoption name EvalFile value ", 30)) { if (!load_net(line + 30)) printf("info string cannot load %s\n", line + 30); }
        else if (!strncmp(line, "setoption name Hash value ", 26)) tt_alloc(atoi(line + 26));
        else if (!strcmp(line, "ucinewgame")) { memset(tt, 0, (tt_mask + 1) * sizeof(TTE)); memset(history_h, 0, sizeof history_h); }
        else if (!strncmp(line, "position", 8)) {
            char *p = line + 9;
            if (!strncmp(p, "startpos", 8)) set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            else if (!strncmp(p, "fen ", 4)) set_fen(p + 4);
            char *mv = strstr(p, "moves");
            if (mv) {
                mv += 5;
                for (;;) {
                    while (*mv == ' ') mv++;
                    if (!*mv) break;
                    Move m = parse_move(mv);
                    if (!m || !make(m)) break;
                    while (*mv && *mv != ' ') mv++;
                }
            }
        } else if (!strncmp(line, "bonus", 5) || !strncmp(line, "setoption name Bonus value", 26)) {   // bonus e2e4 12 d2d4 -5 ...  (next go only)
            memset(root_bonus, 0, sizeof root_bonus); use_bonus = 1;
            char *p = line + (line[0] == 'b' ? 5 : 26);
            for (;;) {
                while (*p == ' ') p++;
                if (!*p) break;
                int f = (p[0] - 'a') + 8 * (p[1] - '1'), t = (p[2] - 'a') + 8 * (p[3] - '1');
                while (*p && *p != ' ') p++;
                int v = atoi(p);
                while (*p == ' ') p++;
                while (*p && *p != ' ') p++;
                if (f >= 0 && f < 64 && t >= 0 && t < 64) root_bonus[f * 64 + t] = v;
            }
        } else if (!strncmp(line, "go", 2)) {
            int depth = 64, movetime = 0, wtime = -1, btime = -1, winc = 0, binc = 0;
            node_limit = 0; print_root = 0;
            char *p;
            if ((p = strstr(line, "depth"))) depth = atoi(p + 6);
            if ((p = strstr(line, "movetime"))) movetime = atoi(p + 9);
            if ((p = strstr(line, "nodes"))) node_limit = strtoull(p + 6, 0, 10);
            if ((p = strstr(line, "wtime"))) wtime = atoi(p + 6);
            if ((p = strstr(line, "btime"))) btime = atoi(p + 6);
            if ((p = strstr(line, "winc"))) winc = atoi(p + 5);
            if ((p = strstr(line, "binc"))) binc = atoi(p + 5);
            if (strstr(line, "rootscores")) print_root = 1;
            time_ms_limit = hard_ms_limit = 0;
            if (movetime) { hard_ms_limit = movetime > 12 ? movetime - 6 : movetime; time_ms_limit = hard_ms_limit * 6 / 10; }
            else if (wtime >= 0 || btime >= 0) {
                int t = P.side == WHITE ? wtime : btime, inc = P.side == WHITE ? winc : binc;
                int alloc = t / 30 + inc * 3 / 4; if (alloc > t / 3) alloc = t / 3; if (alloc < 5) alloc = 5;
                time_ms_limit = alloc * 6 / 10; hard_ms_limit = alloc * 2; if (hard_ms_limit > t / 2) hard_ms_limit = t / 2;
            }
            if (depth > 64) depth = 64;
            nodes = 0; stop_flag = 0; clock_gettime(CLOCK_MONOTONIC, &t_start);
            Move m = search_root(depth);
            if (print_root) {
                printf("info string rootscores");
                for (int i = 0; i < n_root; i++) { char ms[8]; move_str(root_moves[i], ms); printf(" %s %d", ms, root_scores[i]); }
                printf("\n");
            }
            use_bonus = 0;
            char ms[8];
            if (m) { move_str(m, ms); printf("bestmove %s\n", ms); } else printf("bestmove 0000\n");
        } else if (!strncmp(line, "perft", 5)) { printf("perft %llu\n", perft(atoi(line + 6))); }
        else if (!strcmp(line, "eval")) printf("eval %d\n", evaluate());
        else if (!strncmp(line, "see ", 4)) { Move m = parse_move(line + 4); printf("see %d\n", m ? see(m) : -99999); }
        else if (!strcmp(line, "quit")) break;
        fflush(stdout);
    }
    return 0;
}
