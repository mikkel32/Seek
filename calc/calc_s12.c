// CALC — the fast "calculator" half of the two-system engine.
// Bitboard move generation, PVS alpha-beta with a transposition table, null move, late-move reductions,
// quiescence with static exchange evaluation, and a small NNUE-style evaluation read from a weights file.
// Speaks a subset of UCI. Custom commands: perft N, eval, bonus <move> <cp> ... (root prior from OURO).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

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
static inline U64 bishop_slow(int s, U64 occ) { return ray_pos(D_NE, s, occ) | ray_pos(D_NW, s, occ) | ray_neg(D_SE, s, occ) | ray_neg(D_SW, s, occ); }
static inline U64 rook_slow(int s, U64 occ) { return ray_pos(D_N, s, occ) | ray_pos(D_E, s, occ) | ray_neg(D_S, s, occ) | ray_neg(D_W, s, occ); }
static U64 rmask[64], bmask[64], rstore[102400], bstore[5248], *rtab[64], *btab[64];
#ifdef NO_PEXT                           /* AMD Zen 1/2 run PEXT in microcode: scan the rays instead */
static inline U64 rook_att(int s, U64 occ) { return rook_slow(s, occ); }
static inline U64 bishop_att(int s, U64 occ) { return bishop_slow(s, occ); }
#else
static inline U64 rook_att(int s, U64 occ) { return rtab[s][_pext_u64(occ, rmask[s])]; }
static inline U64 bishop_att(int s, U64 occ) { return btab[s][_pext_u64(occ, bmask[s])]; }
#endif
static void init_pext(void) {
    U64 *rp = rstore, *bp = bstore;
    for (int s = 0; s < 64; s++) {
        U64 rm = 0, bm = 0;
        for (int d = 0; d < 8; d++) {                       // a ray without its last square: the edge never blocks anything
            U64 r = rays[d][s];
            if (r) r &= ~BIT(d < 4 ? msb(r) : lsb(r));
            if (d == D_N || d == D_E || d == D_S || d == D_W) rm |= r; else bm |= r;
        }
        rmask[s] = rm; bmask[s] = bm; rtab[s] = rp; btab[s] = bp;
        U64 sub = 0; do { rp[_pext_u64(sub, rm)] = rook_slow(s, sub); sub = (sub - rm) & rm; } while (sub);
        sub = 0; do { bp[_pext_u64(sub, bm)] = bishop_slow(s, sub); sub = (sub - bm) & bm; } while (sub);
        rp += 1ULL << popc(rm); bp += 1ULL << popc(bm);
    }
}

// ------------------------------------------------------------------ position
typedef struct { U64 bb[2][6], occ[2]; signed char sq[64]; int side, castle, ep, half; U64 key, pkey, npkey[2]; } Pos;
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
// evaluation accumulators, one per history ply, kept for both viewpoints and updated incrementally by make()
#ifndef HMAX
#define HMAX 512                         // widest hidden layer this build can load (-DHMAX=... for wider experiments)
#endif
// One accumulator per history ply. Updates are LAZY: make() only records what changed; the sums are built from the
// nearest computed ancestor when (and only if) a position is actually evaluated.
typedef struct { short a[2][HMAX] __attribute__((aligned(32))); float lin[2][2]; int pb, ok, nchg, ksq[2]; int chg[6][4]; } Acc;   // sums as 16-bit integers (see quantize_net)
static Acc *accs;
static int NH = 0;                       // hidden width; 0 = linear piece-square evaluation only
#define NBMAX 32
static int NBK = 1;                      // piece-count groups of output experts
static int NM = 1, NG = 0;               // experts per group (chosen by a learned router) and router inputs per side
// big tables (hash table, input weights, accumulator stack) on 2 MB pages where the system allows it: far fewer TLB misses
#include <sys/mman.h>
static void *big_alloc(size_t bytes) {
    size_t sz = (bytes + ((size_t)2 << 20) - 1) & ~(((size_t)2 << 20) - 1);
    void *p = aligned_alloc((size_t)2 << 20, sz);
    if (!p) { p = aligned_alloc(64, (bytes + 63) / 64 * 64); sz = (bytes + 63) / 64 * 64; }
#ifdef MADV_HUGEPAGE
    else madvise(p, sz, MADV_HUGEPAGE);
#endif
    memset(p, 0, sz);
    return p;
}
static float NGW[NBMAX][128], NGB[NBMAX];
static float *NE, NB[HMAX], NWO[NBMAX][HMAX], NWP[NBMAX][HMAX], NCB[NBMAX], NC = 0, NS[768], NC2 = 0, NS2[768];
static float NQO[NBMAX][HMAX / 2], NQP[NBMAX][HMAX / 2], NQX[NBMAX][HMAX]; static int NPROD = 0;   // product terms (see evaluate)
// The network in integers: hidden-layer sums as 16-bit numbers in units of 2^-q_bits, output weights in units of 2^-11.
// Sixteen sums fit in one AVX2 register instead of eight, the input table is half the size, and integer sums are exact:
// adding and removing pieces along a search path can never drift.
static short *NEq, NBq[HMAX], NWOq[NBMAX][HMAX], NWPq[NBMAX][HMAX], NQOq[NBMAX][HMAX / 2], NQPq[NBMAX][HMAX / 2], NQXq[NBMAX][HMAX];
static int q_bits = 8;
// reliability read-out (optional trailer 'VOL1'): predicts how far a search will move the static score of this position
// "where to search" read-out (optional trailer 'POL1', zero/pol_train.py): from the hidden layer, a score for every quiet move
#define PHMAX 64
static float *PW1, PB1[PHMAX], PW2[448][PHMAX], PB2[448]; static int NPH = 0, have_pol = 0;
static float NV[NBMAX][2 * HMAX + 1]; static short NVOq[NBMAX][HMAX], NVPq[NBMAX][HMAX]; static int have_vol = 0;
static int have_net = 0, taper = 0;   // taper: second (endgame) linear table, blended by remaining material
static int NKB = 1;                      // king buckets for the hidden-layer inputs (1 = none); mirrored so the king is on files a-d
static const int KBT[32] = {0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
// 11 un-mirrored buckets by own-king square (own side moving up): back rank in six groups, second rank in three, then two bands
static const int KB11[64] = {0, 0, 1, 2, 3, 4, 5, 5,  6, 6, 6, 7, 7, 8, 8, 8,  9, 9, 9, 9, 9, 9, 9, 9,  9, 9, 9, 9, 9, 9, 9, 9,
                             10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
static inline int feat_ix(int v, int c, int pt, int s) { return ((c == v ? 0 : 6) + pt) * 64 + (v == WHITE ? s : s ^ 56); }
static inline int feat_fx(int pb, int v, int c, int pt, int s, int ksq) {       // row of the input matrix (depends on the bucket scheme)
    int ix = feat_ix(v, c, pt, s);
    if (NKB == 1) return ix;
    if (NKB == 8) return pb * 768 + ix;
    if (NKB == 11) return KB11[v == WHITE ? ksq : ksq ^ 56] * 768 + ix;
    int sv = v == WHITE ? s : s ^ 56, kv = v == WHITE ? ksq : ksq ^ 56, mir = (kv & 7) >= 4 ? 7 : 0, kq = kv ^ mir;
    return KBT[(kq >> 3) * 4 + (kq & 7)] * 768 + ((c == v ? 0 : 6) + pt) * 64 + (sv ^ mir);
}
// add (sign=+1) or remove (-1) one piece in one viewpoint's accumulator; ksq = that viewpoint's own king square (board coordinates)
static inline void acc_apply(Acc *A, int v, int sign, int c, int pt, int s, int ksq) {
    int sv = v == WHITE ? s : s ^ 56, base = ((c == v ? 0 : 6) + pt) * 64, ix = base + sv, fx = ix;
    if (NKB == 8) fx = A->pb * 768 + ix;                                  // eight phase specialists, chosen by piece count
    else if (NKB == 11) fx = KB11[v == WHITE ? ksq : ksq ^ 56] * 768 + ix;
    else if (NKB > 1) {
        int kv = v == WHITE ? ksq : ksq ^ 56, mir = (kv & 7) >= 4 ? 7 : 0, kq = kv ^ mir;
        fx = KBT[(kq >> 3) * 4 + (kq & 7)] * 768 + base + (sv ^ mir);
    }
    const short *e = NEq + (size_t)fx * NH; short *a = A->a[v];
    if (sign > 0) { for (int i = 0; i < NH; i++) a[i] += e[i]; A->lin[v][0] += NS[ix]; A->lin[v][1] += NS2[ix]; }
    else { for (int i = 0; i < NH; i++) a[i] -= e[i]; A->lin[v][0] -= NS[ix]; A->lin[v][1] -= NS2[ix]; }
}
static inline void put(int c, int pt, int s) { P.bb[c][pt] |= BIT(s); P.occ[c] |= BIT(s); P.sq[s] = c * 6 + pt; P.key ^= zpiece[c * 6 + pt][s]; if (pt == PAWN) P.pkey ^= zpiece[c * 6][s]; else P.npkey[c] ^= zpiece[c * 6 + pt][s]; }
static inline void rem(int c, int pt, int s) { P.bb[c][pt] ^= BIT(s); P.occ[c] ^= BIT(s); P.sq[s] = EMPTY; P.key ^= zpiece[c * 6 + pt][s]; if (pt == PAWN) P.pkey ^= zpiece[c * 6][s]; else P.npkey[c] ^= zpiece[c * 6 + pt][s]; }

static inline int phase_bucket(void) { int b = (popc(P.occ[0] | P.occ[1]) - 2) / 4; return b > 7 ? 7 : b; }
static void acc_refresh_view(int v) {
    Acc *A = &accs[hply];
    A->pb = phase_bucket();
    int ksq = lsb(P.bb[v][KING]);
    memcpy(A->a[v], NBq, sizeof(short) * NH); A->lin[v][0] = A->lin[v][1] = 0;
    for (int c = 0; c < 2; c++) for (int pt = 0; pt < 6; pt++) { U64 b = P.bb[c][pt]; while (b) { int s = lsb(b); b &= b - 1; acc_apply(A, v, 1, c, pt, s, ksq); } }
}
static void acc_refresh(void) { if (have_net) { acc_refresh_view(WHITE); acc_refresh_view(BLACK); accs[hply].ok = 1; } }
// board changes of the move being made
static int chg_n, chg[6][4];
static inline void putA(int c, int pt, int s) { put(c, pt, s); chg[chg_n][0] = 1; chg[chg_n][1] = c; chg[chg_n][2] = pt; chg[chg_n][3] = s; chg_n++; }
static inline void remA(int c, int pt, int s) { rem(c, pt, s); chg[chg_n][0] = -1; chg[chg_n][1] = c; chg[chg_n][2] = pt; chg[chg_n][3] = s; chg_n++; }
// make() calls this: remember the change, compute nothing. Moves that switch a bucket (rare) rebuild the sums right away,
// because that needs the board as it is now.
static inline void acc_record(int king_moved) {
    if (!have_net) return;
    Acc *A = &accs[hply];
    if (NKB > 1 && ((NKB == 8) ? (phase_bucket() != accs[hply - 1].pb) : king_moved)) { acc_refresh(); return; }
    A->ok = 0; A->nchg = chg_n; A->pb = accs[hply - 1].pb;
    A->ksq[0] = lsb(P.bb[0][KING]); A->ksq[1] = lsb(P.bb[1][KING]);
    for (int i = 0; i < chg_n; i++) { A->chg[i][0] = chg[i][0]; A->chg[i][1] = chg[i][1]; A->chg[i][2] = chg[i][2]; A->chg[i][3] = chg[i][3]; }
}
// build accs[j] from the computed accs[j-1] in ONE pass over the data (no copy first)
static void acc_build(int j) {
    Acc *A = &accs[j]; const Acc *S = &accs[j - 1];
    int n = NH, nc = A->nchg;
    for (int v = 0; v < 2; v++) {
        const short *src = S->a[v]; short *dst = A->a[v];
        const short *r[6]; float sg[6]; float l0 = S->lin[v][0], l1 = S->lin[v][1];
        for (int i = 0; i < nc; i++) {
            int c = A->chg[i][1], pt = A->chg[i][2], sq = A->chg[i][3], ix = feat_ix(v, c, pt, sq);
            r[i] = NEq + (size_t)feat_fx(A->pb, v, c, pt, sq, A->ksq[v]) * n; sg[i] = (float)A->chg[i][0];
            l0 += sg[i] * NS[ix]; l1 += sg[i] * NS2[ix];
        }
        A->lin[v][0] = l0; A->lin[v][1] = l1;
        if (nc == 2) { const short *p = sg[0] > 0 ? r[0] : r[1], *m = sg[0] > 0 ? r[1] : r[0]; for (int i = 0; i < n; i++) dst[i] = src[i] + p[i] - m[i]; }
        else if (nc == 3) {                                               // capture: two pieces leave, one arrives
            const short *p = 0, *m[2]; int k = 0;
            for (int i = 0; i < 3; i++) { if (sg[i] > 0) p = r[i]; else m[k++] = r[i]; }
            for (int i = 0; i < n; i++) dst[i] = src[i] + p[i] - m[0][i] - m[1][i];
        } else if (nc == 0) memcpy(dst, src, sizeof(short) * n);          // null move
        else {
            memcpy(dst, src, sizeof(short) * n);
            for (int k = 0; k < nc; k++) { const short *e = r[k]; if (sg[k] > 0) for (int i = 0; i < n; i++) dst[i] += e[i]; else for (int i = 0; i < n; i++) dst[i] -= e[i]; }
        }
    }
    A->ok = 1;
}
static inline void acc_ensure(void) {
    int j = hply;
    while (!accs[j].ok) j--;
    for (j++; j <= hply; j++) acc_build(j);
}
static int attacked(int s, int by) {
    if (pawn_att[by ^ 1][s] & P.bb[by][PAWN]) return 1;
    if (knight_att[s] & P.bb[by][KNIGHT]) return 1;
    if (king_att[s] & P.bb[by][KING]) return 1;
    U64 occ = P.occ[0] | P.occ[1];
    if (bishop_att(s, occ) & (P.bb[by][BISHOP] | P.bb[by][QUEEN])) return 1;
    if (rook_att(s, occ) & (P.bb[by][ROOK] | P.bb[by][QUEEN])) return 1;
    return 0;
}
static U64 attacks_by(int c) {
    U64 occ = P.occ[0] | P.occ[1], a, b = P.bb[c][PAWN];
    const U64 FA = 0x0101010101010101ULL, FH = 0x8080808080808080ULL;
    a = c == WHITE ? (((b & ~FA) << 7) | ((b & ~FH) << 9)) : (((b & ~FA) >> 9) | ((b & ~FH) >> 7));
    for (b = P.bb[c][KNIGHT]; b; b &= b - 1) a |= knight_att[lsb(b)];
    for (b = P.bb[c][BISHOP] | P.bb[c][QUEEN]; b; b &= b - 1) a |= bishop_att(lsb(b), occ);
    for (b = P.bb[c][ROOK] | P.bb[c][QUEEN]; b; b &= b - 1) a |= rook_att(lsb(b), occ);
    return a | king_att[lsb(P.bb[c][KING])];
}
static inline int in_check(void) { return attacked(lsb(P.bb[P.side][KING]), P.side ^ 1); }

// transposition table (declared here so make() can prefetch the entry of the position it reaches)
typedef struct { U64 key; Move move; short score; unsigned char depth, flag; } TTE;
static TTE *tt; static U64 tt_mask;
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
    chg_n = 0;
    if (fl == F_EP) remA(them, PAWN, t + (us == WHITE ? -8 : 8));
    else if (cap != EMPTY) remA(them, cap % 6, t);
    remA(us, pt, f); putA(us, pr ? pr : pt, t);
    if (fl == F_CASTLE) { if (t > f) { remA(us, ROOK, f + 3); putA(us, ROOK, f + 1); } else { remA(us, ROOK, f - 4); putA(us, ROOK, f - 1); } }
    acc_record(pt == KING);
    if (P.ep != NOSQ) P.key ^= zep[P.ep & 7];
    P.ep = NOSQ;
    if (fl == F_DPUSH && (pawn_att[us][(f + t) / 2] & P.bb[them][PAWN])) { P.ep = (f + t) / 2; P.key ^= zep[P.ep & 7]; }
    P.key ^= zcastle[P.castle]; P.castle &= cmask[f] & cmask[t]; P.key ^= zcastle[P.castle];
    P.half = (pt == PAWN || cap != EMPTY || fl == F_EP) ? 0 : P.half + 1;
    P.side = them; P.key ^= zside;
    if (attacked(lsb(P.bb[us][KING]), them)) { unmake(); return 0; }
    __builtin_prefetch(&tt[P.key & tt_mask]);
    return 1;
}
static void make_null(void) {
    Undo *u = &ust[hply]; u->m = 0; u->cap = EMPTY; u->castle = P.castle; u->ep = P.ep; u->half = P.half; u->key = P.key;
    keys[hply++] = P.key;
    chg_n = 0; acc_record(0);
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
    int mode = caps_only; caps_only = mode == 1 || mode == 3;
    U64 target = mode == 2 ? empty : (caps_only ? opp : ~own);
    U64 pawns = P.bb[us][PAWN];
    int up = us == WHITE ? 8 : -8;
    U64 r7 = us == WHITE ? 0x00FF000000000000ULL : 0x000000000000FF00ULL, r3 = us == WHITE ? 0x0000000000FF0000ULL : 0x0000FF0000000000ULL;
    U64 b = pawns;
    while (b) {
        int f = lsb(b); b &= b - 1;
        int promo = (BIT(f) & r7) != 0;
        U64 a = mode == 2 ? 0 : pawn_att[us][f] & opp;
        while (a) { int t = lsb(a); a &= a - 1; if (promo) l = add_promos(l, f, t, 1); else *l++ = MK(f, t, 0, 0); }
        if (mode != 2 && P.ep != NOSQ && (pawn_att[us][f] & BIT(P.ep))) *l++ = MK(f, P.ep, 0, F_EP);
        int t = f + up;
        if (empty & BIT(t)) {
            if (promo) { if (mode != 2) l = add_promos(l, f, t, mode != 1); }
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
static int mat_val[7] = {100, 320, 330, 500, 950, 0, 0};
static int see_val[7] = {100, 320, 330, 500, 950, 20000, 0};
static int eval_mode = 0;                // 1 = rules-only evaluation (mobility)
static void set_piece_values(const int *v) { for (int i = 0; i < 5; i++) { mat_val[i] = v[i]; see_val[i] = v[i]; } }

static void quantize_net(void) {
    // The largest sum any position could produce, per hidden unit: its bias plus the 32 largest per-square maxima (a square
    // holds one piece, a board 32 pieces). The integer scale is the finest power of two at which that still fits 16 bits.
    float worst = 0;
    for (int i = 0; i < NH; i++) {
        float m[64], sum = fabsf(NB[i]);
        for (int sq = 0; sq < 64; sq++) { float x = 0; for (int r = 0; r < NKB * 12; r++) { float v = fabsf(NE[((size_t)r * 64 + sq) * NH + i]); if (v > x) x = v; } m[sq] = x; }
        for (int k = 0; k < 32; k++) { int b = k; for (int j = k + 1; j < 64; j++) if (m[j] > m[b]) b = j; float t = m[b]; m[b] = m[k]; m[k] = t; sum += t; }
        if (sum > worst) worst = sum;
    }
    q_bits = 10; while (q_bits > 4 && worst * (float)(1 << q_bits) > 32000.0f) q_bits--;
    float qa = (float)(1 << q_bits); size_t ne = (size_t)NKB * 768 * NH;
    free(NEq); NEq = (short *)big_alloc(ne * 2);
    for (size_t k = 0; k < ne; k++) NEq[k] = (short)lrintf(NE[k] * qa);
    for (int i = 0; i < NH; i++) NBq[i] = (short)lrintf(NB[i] * qa);
    #define QW(x) ({ float t_ = (x) * 2048.0f; (short)lrintf(t_ > 16383.0f ? 16383.0f : (t_ < -16383.0f ? -16383.0f : t_)); })   /* output weights beyond +-8 are clipped */
    for (int k = 0; k < NBMAX; k++) {
        for (int i = 0; i < NH; i++) { NWOq[k][i] = QW(NWO[k][i]); NWPq[k][i] = QW(NWP[k][i]); NQXq[k][i] = NPROD ? QW(NQX[k][i]) : 0; }
        for (int i = 0; i < NH / 2; i++) { NQOq[k][i] = NPROD ? QW(NQO[k][i]) : 0; NQPq[k][i] = NPROD ? QW(NQP[k][i]) : 0; }
        for (int i = 0; i < NH; i++) { NVOq[k][i] = have_vol ? QW(NV[k][i]) : 0; NVPq[k][i] = have_vol ? QW(NV[k][NH + i]) : 0; }
    }
    #undef QW
}
static int load_net(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int hdr[2], nb = 1, nk = 1, nm = 1, ng = 0;
    if (fread(hdr, 4, 2, f) != 2 || (hdr[0] != 0x4F55524F && hdr[0] != 0x4F555232 && hdr[0] != 0x4F555233 && hdr[0] != 0x4F555235 && hdr[0] != 0x4F555236) || hdr[1] < 0 || hdr[1] > HMAX) { fclose(f); return 0; }
    if (hdr[0] != 0x4F55524F && (fread(&nb, 4, 1, f) != 1 || nb < 1 || nb > NBMAX)) { fclose(f); return 0; }
    if ((hdr[0] == 0x4F555233 || hdr[0] == 0x4F555235 || hdr[0] == 0x4F555236) && (fread(&nk, 4, 1, f) != 1 || nk < 1 || nk > 16)) { fclose(f); return 0; }
    if (hdr[0] == 0x4F555235 && (fread(&nm, 4, 1, f) != 1 || fread(&ng, 4, 1, f) != 1 || nm < 1 || nb % nm || ng < 1 || ng > 64)) { fclose(f); return 0; }
    have_net = 0; NH = hdr[1]; NBK = nb / nm; NKB = nk; NM = nm; NG = ng;
    free(NE); NE = (float *)big_alloc((size_t)nk * 768 * (NH ? NH : 1) * 4);
    size_t ok = 1;
    memset(NCB, 0, sizeof NCB);
    if (NH) {
        ok &= fread(NE, 4, (size_t)nk * 768 * NH, f) == (size_t)nk * 768 * NH; ok &= fread(NB, 4, NH, f) == (size_t)NH;
        for (int k = 0; k < nb; k++) { ok &= fread(NWO[k], 4, NH, f) == (size_t)NH; ok &= fread(NWP[k], 4, NH, f) == (size_t)NH; }
    }
    ok &= fread(&NC, 4, 1, f) == 1; ok &= fread(NS, 4, 768, f) == 768;
    taper = ok && fread(&NC2, 4, 1, f) == 1 && fread(NS2, 4, 768, f) == 768;
    if (hdr[0] != 0x4F55524F) ok &= fread(NCB, 4, nb, f) == (size_t)nb;
    if (nm > 1) { for (int k = 0; k < nb; k++) ok &= fread(NGW[k], 4, 2 * ng, f) == (size_t)(2 * ng); ok &= fread(NGB, 4, nb, f) == (size_t)nb; }
    NPROD = hdr[0] == 0x4F555236 && NH;
    if (NPROD) for (int k = 0; k < nb; k++) { ok &= fread(NQO[k], 4, NH / 2, f) == (size_t)(NH / 2); ok &= fread(NQP[k], 4, NH / 2, f) == (size_t)(NH / 2); ok &= fread(NQX[k], 4, NH, f) == (size_t)NH; }
    have_vol = 0; have_pol = 0;
    for (int tag = 0; ok && fread(&tag, 4, 1, f) == 1; ) {                      // optional trailers, in any order
        if (tag == 0x4C415650) { int pv[5]; if (fread(pv, 4, 5, f) == 5) set_piece_values(pv); else break; }
        else if (tag == 0x314C4F56 && nm == 1 && NH) { int good = 1; for (int k = 0; k < nb; k++) good &= fread(NV[k], 4, 2 * NH + 1, f) == (size_t)(2 * NH + 1); have_vol = good; if (!good) break; }
        else if (tag == 0x314C4F50 && nm == 1 && NH) {
            int hid = 0, good = fread(&hid, 4, 1, f) == 1 && hid >= 1 && hid <= PHMAX;
            if (!good) break;
            free(PW1); PW1 = (float *)malloc((size_t)hid * 2 * NH * 4);
            good &= fread(PW1, 4, (size_t)hid * 2 * NH, f) == (size_t)hid * 2 * NH; good &= fread(PB1, 4, hid, f) == (size_t)hid;
            for (int k = 0; k < 448; k++) good &= fread(PW2[k], 4, hid, f) == (size_t)hid;
            good &= fread(PB2, 4, 448, f) == 448; NPH = hid; have_pol = good; if (!good) break;
        }
        else break;
    }
    fclose(f);
    if (!taper) { NC2 = 0; memset(NS2, 0, sizeof NS2); }
    have_net = (int)ok;
    if (have_net && NH) quantize_net();
    acc_refresh();
    return have_net;
}
// sum_i w[i] * clamp(a[i], 0, 1)^2 — explicit vector code: a plain loop adds the terms one after another
static inline float screlu_dot(const float *a, const float *w, int n) {
    int i = 0; float out = 0;
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps(); const __m512 zero = _mm512_setzero_ps(), one = _mm512_set1_ps(1.0f);
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_min_ps(_mm512_max_ps(_mm512_loadu_ps(a + i), zero), one);
        acc = _mm512_fmadd_ps(_mm512_mul_ps(x, x), _mm512_loadu_ps(w + i), acc);
    }
    out = _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 acc = _mm256_setzero_ps(); const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(a + i), zero), one);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(x, x), _mm256_loadu_ps(w + i), acc);
    }
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo); out = _mm_cvtss_f32(lo);
#endif
    for (; i < n; i++) { float x = a[i] < 0 ? 0 : (a[i] > 1 ? 1 : a[i]); out += w[i] * x * x; }
    return out;
}
// sum_i w[i] * clamp(a[i], 0, 1) * clamp(b[i], 0, 1): two different hidden units multiplied. A square can only say "this
// pattern, strongly"; a product says "this pattern AND that one" (two of my own, or one of mine and one of the opponent's).
static inline float prod_dot(const float *a, const float *b, const float *w, int n) {
    int i = 0; float out = 0;
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps(); const __m512 zero = _mm512_setzero_ps(), one = _mm512_set1_ps(1.0f);
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_min_ps(_mm512_max_ps(_mm512_loadu_ps(a + i), zero), one), y = _mm512_min_ps(_mm512_max_ps(_mm512_loadu_ps(b + i), zero), one);
        acc = _mm512_fmadd_ps(_mm512_mul_ps(x, y), _mm512_loadu_ps(w + i), acc);
    }
    out = _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 acc = _mm256_setzero_ps(); const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(a + i), zero), one), y = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(b + i), zero), one);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(x, y), _mm256_loadu_ps(w + i), acc);
    }
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo); out = _mm_cvtss_f32(lo);
#endif
    for (; i < n; i++) { float x = a[i] < 0 ? 0 : (a[i] > 1 ? 1 : a[i]), y = b[i] < 0 ? 0 : (b[i] > 1 ? 1 : b[i]); out += w[i] * x * y; }
    return out;
}
// sum_i w[i] * clamp(a[i], 0, 1) * clamp(b[i], 0, 1) in integers (a == b gives the squares). The clamped values, shifted to
// 15 bits, are multiplied keeping the high half: a 14-bit product; times an 11-bit-scaled weight the result is in units of 2^-25.
static inline long long qdot(const short *a, const short *b, const short *w, int n) {
    long long out = 0; int i = 0; const int one_i = 1 << q_bits, sh_i = 15 - q_bits;
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256(), one = _mm256_set1_epi16((short)one_i); const __m128i sh = _mm_cvtsi32_si128(sh_i);
    __m256i acc64 = _mm256_setzero_si256();
    while (i + 16 <= n) {
        __m256i acc32 = _mm256_setzero_si256();
        for (int k = 0; k < 4 && i + 16 <= n; k++, i += 16) {          // four steps at most before widening: cannot overflow 32 bits
            __m256i x = _mm256_sll_epi16(_mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256((const __m256i *)(a + i)), zero), one), sh);
            __m256i y = _mm256_sll_epi16(_mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256((const __m256i *)(b + i)), zero), one), sh);
            acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(_mm256_mulhi_epu16(x, y), _mm256_loadu_si256((const __m256i *)(w + i))));
        }
        acc64 = _mm256_add_epi64(acc64, _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(acc32)), _mm256_cvtepi32_epi64(_mm256_extracti128_si256(acc32, 1))));
    }
    long long t[4]; _mm256_storeu_si256((__m256i *)t, acc64); out = t[0] + t[1] + t[2] + t[3];
#endif
    for (; i < n; i++) {
        unsigned x = (unsigned)(a[i] < 0 ? 0 : (a[i] > one_i ? one_i : a[i])) << sh_i, y = (unsigned)(b[i] < 0 ? 0 : (b[i] > one_i ? one_i : b[i])) << sh_i;
        out += (long long)((x * y) >> 16) * w[i];
    }
    return out;
}
static int evaluate(void) {
    int us = P.side;
    if (eval_mode == 1) {                // rules only: how many moves do I have, minus how many you would have
        Move l[256];
        int a = gen(l, 0); P.side ^= 1; int b = gen(l, 0); P.side ^= 1;
        return 10 * (a - b);
    }
    if (!have_net) {                     // fallback: bare material, so the engine runs without a weights file
        int v = 0;
        for (int pt = 0; pt < 5; pt++) v += mat_val[pt] * (popc(P.bb[us][pt]) - popc(P.bb[us ^ 1][pt]));
        return v;
    }
    acc_ensure();
    const Acc *A = &accs[hply];
    const short *ao = A->a[us], *ap = A->a[us ^ 1];
    int n = NH;
    float lin = NC + A->lin[us][0];
    if (taper) {
        int npm = popc(P.bb[0][KNIGHT] | P.bb[1][KNIGHT] | P.bb[0][BISHOP] | P.bb[1][BISHOP]) + 2 * popc(P.bb[0][ROOK] | P.bb[1][ROOK]) + 4 * popc(P.bb[0][QUEEN] | P.bb[1][QUEEN]);
        float ph = npm >= 24 ? 1.0f : npm / 24.0f;
        lin = ph * lin + (1.0f - ph) * (NC2 + A->lin[us][1]);
    }
    int bk = 0;
    if (NBK > 1) { bk = (popc(P.occ[0] | P.occ[1]) - 2) / 4; if (bk > NBK - 1) bk = NBK - 1; }
    if (NM > 1) {                        // learned router: the piece-count group has NM specialists; run the one the router scores highest
        float x[128]; int best = 0; float bg = -1e30f;
        const float qi = 1.0f / (float)(1 << q_bits);
        for (int i = 0; i < NG; i++) { float a = ao[i] * qi, b = ap[i] * qi; a = a < 0 ? 0 : (a > 1 ? 1 : a); b = b < 0 ? 0 : (b > 1 ? 1 : b); x[i] = a * a; x[NG + i] = b * b; }
        for (int m = 0; m < NM; m++) {
            const float *gw = NGW[bk * NM + m]; float g = NGB[bk * NM + m];
            for (int i = 0; i < 2 * NG; i++) g += gw[i] * x[i];
            if (g > bg) { bg = g; best = m; }
        }
        bk = bk * NM + best;
    }
    long long q = qdot(ao, ao, NWOq[bk], n) + qdot(ap, ap, NWPq[bk], n);
    if (NPROD) q += qdot(ao, ao + n / 2, NQOq[bk], n / 2) + qdot(ap, ap + n / 2, NQPq[bk], n / 2) + qdot(ao, ap, NQXq[bk], n);
    float out = lin + NCB[bk] + (float)q * (1.0f / 33554432.0f);
    int v = (int)(out * 271.58f);        // logit of win probability -> centipawns
    return v > 20000 ? 20000 : (v < -20000 ? -20000 : v);
}

// How far is a search likely to move the static score of this position? In centipawns, read off the same hidden layer
// (call after evaluate(): the sums are in place). Lets the search prune with narrow margins where the evaluator is
// reliable and wide ones where it is not.
static int eval_vol(void) {
    const Acc *A = &accs[hply]; int us = P.side, bk = 0;
    if (NBK > 1) { bk = (popc(P.occ[0] | P.occ[1]) - 2) / 4; if (bk > NBK - 1) bk = NBK - 1; }
    float u = NV[bk][2 * NH] + (float)(qdot(A->a[us], A->a[us], NVOq[bk], NH) + qdot(A->a[us ^ 1], A->a[us ^ 1], NVPq[bk], NH)) * (1.0f / 33554432.0f);
    int v = (int)(u * 271.58f);
    return v < 10 ? 10 : (v > 600 ? 600 : v);
}

// ------------------------------------------------------------------ static exchange evaluation
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
enum { TT_NONE, TT_EXACT, TT_LOWER, TT_UPPER };
static void tt_alloc(int mb) { U64 n = 1; while (n * 2 * sizeof(TTE) <= (U64)mb << 20) n *= 2; free(tt); tt = (TTE *)big_alloc(n * sizeof(TTE)); tt_mask = n - 1; }

// ------------------------------------------------------------------ search
// search parameters (hand-set defaults; tunable through UCI options so the engine can tune them by playing itself)
#define NPAR 48
enum { RAZOR, RFP, NMP_BASE, NMP_DIV, NMP_EV, LMP_BASE, FUT_BASE, FUT_MUL, HIST_PRUNE, SEE_Q, SEE_C, SE_DEPTH, SE_MUL, ASP, DELTA_Q, LMR_BASE, LMR_DIV, HB_QUAD, HB_LIN, HB_MAX, PROBCUT, SE_DOUBLE, CUT_LMR, CUT_IIR, NEG_EXT, TTCAP_LMR, LMR_HIST, CAP_LMR, DEEPER, NP_CORR, CHK_EXT, VOL_MIX, VOL_REF, VOL_LMR, VOL_NMP, CONT_CORR, NMP_TT, PV_PRUNE, THR_HIST, VOL_TM, ROOT_LMR, ROOT_ORD, POL_W, POL_DEPTH, POL_LMR, TT_BUCKET, LEARN_MIX, LEARN_RATE };
static const char *par_name[NPAR] = {"Razor", "Rfp", "NmpBase", "NmpDiv", "NmpEv", "LmpBase", "FutBase", "FutMul", "HistPrune", "SeeQ", "SeeC", "SeDepth", "SeMul", "Asp", "DeltaQ", "LmrBase", "LmrDiv", "HbQuad", "HbLin", "HbMax", "ProbCut", "SeDouble", "CutLmr", "CutIir", "NegExt", "TtCapLmr", "LmrHist", "CapLmr", "Deeper", "NpCorr", "ChkExt", "VolMix", "VolRef", "VolLmr", "VolNmp", "ContCorr", "NmpTt", "PvPrune", "ThrHist", "VolTm", "RootLmr", "RootOrd", "PolW", "PolDepth", "PolLmr", "TtBucket", "LearnMix", "LearnRate"};
// search parameters the engine tuned for itself: 20,000 self-play iterations (zero/spsa.py). Against the hand-set values
// (250, 70, 3, 4, 200, 3, 90, 85, 4000, 50, 90, 7, 2, 25, 200, 75, 225, 4, 20, 1200) they scored +50 Elo at 2,500 nodes
// and +37 at 10,000 nodes over 2,000 games each.
static int par[NPAR] = {293, 53, 3, 4, 191, 6, 89, 94, 4245, 48, 92, 7, 2, 33, 190, 68, 240, 5, 33, 1322, 0, 0, 0, 0, 0, 0, 8192, 1, 1, 1, 1, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 32};
static U64 nodes, node_limit, check_mask = 2047;
// (Option TtBucket, off by default: +19 Elo with a 1 MB table at 100,000 positions per move, but -9 at 60+0.6 and -10 at
// 30,000 positions per move with the usual 64 MB, where the table is not under pressure.)
// Hash table with two entries per position slot. A stored position is returned; otherwise the entry to overwrite: one
// left over from an earlier search before one from this search, and of two from the same search the shallower. (With one
// entry per slot, every store destroyed whatever was there: at long time controls the table kept losing its deep results.)
static int tt_age = 0;
static inline TTE *tt_find(U64 key) {
    if (!par[TT_BUCKET]) return &tt[key & tt_mask];
    TTE *b = &tt[key & tt_mask & ~1ULL];
    if (b[0].key == key) return b;
    if (b[1].key == key) return b + 1;
    int v0 = b[0].depth - 8 * ((tt_age - (b[0].flag >> 2)) & 63), v1 = b[1].depth - 8 * ((tt_age - (b[1].flag >> 2)) & 63);
    return v1 < v0 ? b + 1 : b;
}
#define TTF(fl) ((unsigned char)((fl) | (tt_age << 2)))
// self-play harvest: interior positions whose value the search settled exactly (mover-relative features + score)
typedef struct { short idx[32]; short score; signed char result, pad; } Rec;     // 68 bytes, mover's view
static Rec harvest[256]; static int n_harvest, harvest_on;
static void fill_rec(Rec *r, int sc) {
    int k = 0, us = P.side;
    for (int c = 0; c < 2; c++) for (int pt = 0; pt < 6; pt++) {
        U64 b = P.bb[c][pt];
        while (b) { int sq = lsb(b); b &= b - 1; if (k < 32) r->idx[k++] = (short)(((c == us ? 0 : 6) + pt) * 64 + (us == WHITE ? sq : sq ^ 56)); }
    }
    while (k < 32) r->idx[k++] = 768;
    r->score = (short)(sc > 20000 ? 20000 : (sc < -20000 ? -20000 : sc)); r->result = 0; r->pad = 0;
}

static int stop_flag, time_ms_limit, hard_ms_limit, root_depth;
static int tm_new = 1, tm_soft = 55;        // UCI options TmNew, TmSoft (percent of the base share that an average move gets)
static int tm_dynamic;                    // clock games: the time for a move stretches and shrinks with how settled the choice is
static U64 root_move_nodes[4096];         // positions searched under each root move (from*64+to), over the whole search
static struct timespec t_start;
static Move killers[MAXPLY][2];
// Quiet-move history, kept separately for whether the square the piece leaves and the square it goes to are attacked by the
// opponent: "this move was good" means something different when it rescues a piece or walks into an attack.
static int history_t[2][4][64][64];
static U64 thr_stack[MAXPLY + 4];        // squares the opponent attacks, per ply (0 where it was not worked out)
#define HIST(m, ply) history_t[P.side][(int)(((thr_stack[ply] >> FROM(m)) & 1) | (((thr_stack[ply] >> TO(m)) & 1) << 1))][FROM(m)][TO(m)]
static int root_vol = 0;                 // the reliability read-out at the root (for the clock)
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

// per-ply search stack: static evaluation, the move that was made, and the piece that made it
static int evals_stack[MAXPLY + 4];
static Move move_stack[MAXPLY + 4];
static int piece_stack[MAXPLY + 4];
static short cmh[12][64][12][64];          // continuation history: [previous piece][its target][piece][target]
static Move excluded[MAXPLY + 4];          // move left out of a singularity test at this ply
static int cutn[MAXPLY + 8];               // is the node at this ply expected to fail high (set by its parent)
static short fuh[12][64][12][64];          // follow-up history: the same, keyed by our own previous move (two plies back)
static short caph[12][64][6];              // capture history: [piece][target][captured type]
static int corrhist[2][16384];             // correction history: how wrong the static evaluation was for this pawn structure (x256)
static int corr_np[2][2][16384];           // the same, keyed by where one side's pieces (not pawns) stand: [side to move][whose pieces]
static int corr_cont[2][12][64];           // the same, keyed by the move that led to this position: [side to move][piece][target square]
static int corr_ply;                       // the ply of the position being evaluated (set by the callers of eval_corr)
// A read-out that LEARNS DURING THE SEARCH (option LearnMix, percent; 0 = off). The correction tables above remember the
// evaluator's error under a hashed key: a position benefits only from positions with exactly the same pawns, or pieces, or
// previous move. Here the error is predicted from the evaluator's own hidden layer, so a position benefits from every
// position the network itself sees as similar. One weight per hidden unit and side to move, adjusted after every finished
// node by a normalised step towards what the search found (as the tables are): the network's last layer keeps training
// on the tree it is searching. Reset with a new game.
static float FW[2][2 * HMAX]; static short FWq[2][2 * HMAX];
static inline int learn_corr(void) {       // call after evaluate(): the sums are in place
    const Acc *A = &accs[hply]; int us = P.side;
    int v = (int)((float)(qdot(A->a[us], A->a[us], FWq[us], NH) + qdot(A->a[us ^ 1], A->a[us ^ 1], FWq[us] + NH, NH)) * (271.58f / 33554432.0f));
    return v > 48 ? 48 : (v < -48 ? -48 : v);
}
static void learn_update(int err_cp, int w) {
    const Acc *A = &accs[hply]; int us = P.side, n = NH; float x[2 * HMAX], nn = 1e-3f; const float qi = 1.0f / (float)(1 << q_bits);
    if (!A->ok) return;
    for (int i = 0; i < n; i++) {
        float a = A->a[us][i] * qi, b = A->a[us ^ 1][i] * qi; a = a < 0 ? 0 : (a > 1 ? 1 : a); b = b < 0 ? 0 : (b > 1 ? 1 : b);
        x[i] = a * a; x[n + i] = b * b;
    }
    for (int i = 0; i < 2 * n; i++) nn += x[i] * x[i];
    float g = (par[LEARN_RATE] / 256.0f) * (w / 16.0f) * (err_cp / 271.58f) / nn; float *fw = FW[us]; short *fq = FWq[us];
    for (int i = 0; i < 2 * n; i++) {
        float f = fw[i] + g * x[i]; f = f > 7.9f ? 7.9f : (f < -7.9f ? -7.9f : f); fw[i] = f; fq[i] = (short)lrintf(f * 2048.0f);
    }
}
static inline int eval_corr(void) {
    int c = corrhist[P.side][P.pkey & 16383];
    if (par[NP_CORR]) c = (2 * c + corr_np[P.side][0][P.npkey[0] & 16383] + corr_np[P.side][1][P.npkey[1] & 16383]) / 4;
    if (par[CONT_CORR] && corr_ply > 0 && move_stack[corr_ply - 1]) c = (3 * c + corr_cont[P.side][piece_stack[corr_ply - 1]][TO(move_stack[corr_ply - 1])]) / 4;
    int v = evaluate() + c / 256;
    if (par[LEARN_MIX] && have_net) v += learn_corr() * par[LEARN_MIX] / 100;
    return v > 20000 ? 20000 : (v < -20000 ? -20000 : v);
}
static inline int hist_score(Move m, int ply) {
    int h = HIST(m, ply);
    if (ply > 0 && move_stack[ply - 1]) h += cmh[piece_stack[ply - 1]][TO(move_stack[ply - 1])][(int)P.sq[FROM(m)]][TO(m)];
    if (ply > 1 && move_stack[ply - 2]) h += fuh[piece_stack[ply - 2]][TO(move_stack[ply - 2])][(int)P.sq[FROM(m)]][TO(m)];
    return h;
}
static inline void hist_update(Move m, int ply, int bonus) {
    int *h = &HIST(m, ply);
    *h += bonus - *h * abs(bonus) / 16384;
    if (ply > 0 && move_stack[ply - 1]) {
        short *c = &cmh[piece_stack[ply - 1]][TO(move_stack[ply - 1])][(int)P.sq[FROM(m)]][TO(m)];
        *c += bonus - *c * abs(bonus) / 16384;
    }
    if (ply > 1 && move_stack[ply - 2]) {
        short *c = &fuh[piece_stack[ply - 2]][TO(move_stack[ply - 2])][(int)P.sq[FROM(m)]][TO(m)];
        *c += bonus - *c * abs(bonus) / 16384;
    }
}

static void score_moves(Move *list, int *sc, int n, Move ttm, int ply) {
    for (int i = 0; i < n; i++) {
        Move m = list[i];
        if (m == ttm) sc[i] = 2000000000;
        else if (is_capture(m) || PROMO(m)) {
            int victim = FLAG(m) == F_EP ? PAWN : (P.sq[TO(m)] == EMPTY ? NO_PT : P.sq[TO(m)] % 6);
            int v = mat_val[victim] * 16 - mat_val[P.sq[FROM(m)] % 6] / 8 + (PROMO(m) == QUEEN ? 9000 : 0);
            if (victim != NO_PT) v += caph[(int)P.sq[FROM(m)]][TO(m)][victim] / 4;
            sc[i] = (see(m) >= 0 ? 1000000000 : -1000000000) + v;
        } else if (m == killers[ply][0]) sc[i] = 900000000;
        else if (m == killers[ply][1]) sc[i] = 800000000;
        else sc[i] = hist_score(m, ply);
    }
}
static inline Move pick(Move *list, int *sc, int n, int i) {
    int b = i;
    for (int j = i + 1; j < n; j++) if (sc[j] > sc[b]) b = j;
    Move m = list[b]; list[b] = list[i]; list[i] = m;
    int s = sc[b]; sc[b] = sc[i]; sc[i] = s;
    return m;
}

// ------------------------------------------------------------------ staged move picker
// Most nodes end on their first move. Generating, scoring and sorting every move before trying the first one was two
// fifths of the engine's time. Here the table move is tried before anything is generated, captures before the quiet
// moves exist, and a capture's exchange value is worked out only when the capture is about to be tried.
static int pseudo_legal(Move m) {                 // would the generator produce this move in this position?
    int f = FROM(m), t = TO(m), fl = FLAG(m), pr = PROMO(m), us = P.side, them = us ^ 1, pc = P.sq[f];
    if (!m || pc == EMPTY || pc / 6 != us || (P.occ[us] & BIT(t))) return 0;
    int pt = pc % 6; U64 occ = P.occ[0] | P.occ[1];
    if (pt == PAWN) {
        int up = us == WHITE ? 8 : -8, r = f >> 3, last = us == WHITE ? 6 : 1, start = us == WHITE ? 1 : 6;
        if ((r == last) != (pr != 0) || (pr && (pr < KNIGHT || pr > QUEEN))) return 0;
        if (fl == F_EP) return !pr && P.ep != NOSQ && t == P.ep && (pawn_att[us][f] & BIT(t)) != 0;
        if (fl == F_DPUSH) return r == start && t == f + 2 * up && !(occ & (BIT(f + up) | BIT(t)));
        if (fl != F_NORMAL) return 0;
        if (t == f + up) return !(occ & BIT(t));
        return (pawn_att[us][f] & BIT(t) & P.occ[them]) != 0;
    }
    if (pr) return 0;
    if (fl == F_CASTLE) {
        int ks = us == WHITE ? 1 : 4, qs = us == WHITE ? 2 : 8, e = us == WHITE ? 4 : 60;
        if (pt != KING || f != e || attacked(e, them)) return 0;
        if (t == e + 2) return (P.castle & ks) && !(occ & (BIT(e + 1) | BIT(e + 2))) && !attacked(e + 1, them);
        if (t == e - 2) return (P.castle & qs) && !(occ & (BIT(e - 1) | BIT(e - 2) | BIT(e - 3))) && !attacked(e - 1, them);
        return 0;
    }
    if (fl != F_NORMAL) return 0;
    switch (pt) {
        case KNIGHT: return (knight_att[f] & BIT(t)) != 0;
        case BISHOP: return (bishop_att(f, occ) & BIT(t)) != 0;
        case ROOK: return (rook_att(f, occ) & BIT(t)) != 0;
        case QUEEN: return ((bishop_att(f, occ) | rook_att(f, occ)) & BIT(t)) != 0;
        default: return (king_att[f] & BIT(t)) != 0;
    }
}
// sum_i a[i] * b[i]
static inline float fdot(const float *a, const float *b, int n) {
    int i = 0; float out = 0;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo); out = _mm_cvtss_f32(lo);
#endif
    for (; i < n; i++) out += a[i] * b[i];
    return out;
}
// the read-out's hidden units for the position on the board (its evaluation sums must be in place)
static void pol_hidden(float *hid) {
    const Acc *A = &accs[hply]; int us = P.side; float h[2 * HMAX]; const float qi = 1.0f / (float)(1 << q_bits);
    for (int v = 0; v < 2; v++) {
        const short *a = A->a[v ? us ^ 1 : us]; float *d = h + v * NH;
        for (int i = 0; i < NH; i++) { float x = a[i] * qi; x = x < 0 ? 0 : (x > 1 ? 1 : x); d[i] = x * x; }
    }
    for (int k = 0; k < NPH; k++) { float t = PB1[k] + fdot(PW1 + (size_t)k * 2 * NH, h, 2 * NH); hid[k] = t > 0 ? t : 0; }
}
enum { ST_TT, ST_GENCAP, ST_CAP, ST_GENQUIET, ST_QUIET, ST_BAD, ST_DONE };
typedef struct {
    Move list[256], bad[96], ttm; int sc[256], bsc[96], bsee[96];
    int n, i, nbad, stage, ply, caps_only, skip_quiets;
    int pl[256], pol, pol_on, pol_best, pol_now;   // the read-out's score of each quiet move (x256), whether to use it at this node, the best and the current move's
    int hist, bad_now, see_now;                // about the move just returned: its history score (quiet moves), is it a losing capture, its exchange value
} Picker;
static inline void mp_init(Picker *p, Move ttm, int ply, int caps_only) {
    p->ttm = pseudo_legal(ttm) ? ttm : 0; p->stage = p->ttm ? ST_TT : ST_GENCAP; p->ply = ply; p->caps_only = caps_only;
    p->n = p->i = p->nbad = p->skip_quiets = 0; p->pol = p->pol_on = 0;
#ifdef CHECK_PL
    { Move l[256]; int n = gen(l, 0), in = 0; for (int k = 0; k < n; k++) in |= l[k] == ttm; if (ttm && in != (p->ttm != 0)) { printf("pseudo_legal disagrees with the generator: move %d\n", ttm); fflush(stdout); abort(); } }
#endif
}
static Move mp_next(Picker *p) {
    for (;;) switch (p->stage) {
    case ST_TT:
        p->stage = ST_GENCAP; p->bad_now = 0; p->hist = (!is_capture(p->ttm) && !PROMO(p->ttm)) ? hist_score(p->ttm, p->ply) : 0;
        return p->ttm;
    case ST_GENCAP:
        p->n = gen(p->list, p->caps_only ? 1 : 3); p->i = 0;
        for (int k = 0; k < p->n; k++) {
            Move m = p->list[k];
            int victim = FLAG(m) == F_EP ? PAWN : (P.sq[TO(m)] == EMPTY ? NO_PT : P.sq[TO(m)] % 6);
            int v = mat_val[victim] * 16 - mat_val[P.sq[FROM(m)] % 6] / 8 + (PROMO(m) == QUEEN ? 9000 : 0);
            if (victim != NO_PT) v += caph[(int)P.sq[FROM(m)]][TO(m)][victim] / 4;
            p->sc[k] = v;
        }
        p->stage = ST_CAP;
        /* fall through */
    case ST_CAP:
        while (p->i < p->n) {
            int b = p->i;
            for (int j = b + 1; j < p->n; j++) if (p->sc[j] > p->sc[b]) b = j;
            Move m = p->list[b]; int v = p->sc[b]; p->list[b] = p->list[p->i]; p->sc[b] = p->sc[p->i]; p->i++;
            if (m == p->ttm) continue;
            int victim = P.sq[TO(m)] == EMPTY ? NO_PT : P.sq[TO(m)] % 6, sv = 0;
            // taking something worth at least the taker cannot lose material; everything else gets the full exchange calculation
            if (PROMO(m) || FLAG(m) == F_EP || see_val[victim] < see_val[P.sq[FROM(m)] % 6]) sv = see(m);
            if (sv < 0) { if (!p->caps_only && p->nbad < 96) { p->bad[p->nbad] = m; p->bsc[p->nbad] = v; p->bsee[p->nbad++] = sv; } continue; }
            p->bad_now = 0; p->hist = 0;
            return m;
        }
        p->stage = p->caps_only ? ST_DONE : ST_GENQUIET;
        break;
    case ST_GENQUIET:
        if (p->skip_quiets) { p->stage = ST_BAD; p->i = 0; break; }
        p->n = gen(p->list, 2); p->i = 0;
        if (p->pol) {                                   // which quiet moves would a search choose here? (only at nodes with depth left for it to matter)
            float hid[PHMAX], val[448]; unsigned char got[448]; memset(got, 0, sizeof got); pol_hidden(hid);
            int us = P.side; p->pol_best = -1000000; p->pol_on = 1;
            for (int k = 0; k < p->n; k++) {
                Move m = p->list[k]; int f = FROM(m), t = TO(m);
                int a = us == WHITE ? f : f ^ 56, b = 64 + (P.sq[f] % 6) * 64 + (us == WHITE ? t : t ^ 56);
                if (!got[a]) { val[a] = PB2[a] + fdot(PW2[a], hid, NPH); got[a] = 1; }
                if (!got[b]) { val[b] = PB2[b] + fdot(PW2[b], hid, NPH); got[b] = 1; }
                p->pl[k] = (int)((val[a] + val[b]) * 256.0f);
                if (p->pl[k] > p->pol_best) p->pol_best = p->pl[k];
            }
        }
        for (int k = 0; k < p->n; k++) {
            Move m = p->list[k];
            p->sc[k] = m == killers[p->ply][0] ? 900000000 : (m == killers[p->ply][1] ? 800000000 : hist_score(m, p->ply) + (p->pol_on ? p->pl[k] * par[POL_W] / 256 : 0));
        }
        p->stage = ST_QUIET;
        /* fall through */
    case ST_QUIET:
        while (!p->skip_quiets && p->i < p->n) {
            int b = p->i;
            for (int j = b + 1; j < p->n; j++) if (p->sc[j] > p->sc[b]) b = j;
            Move m = p->list[b]; int v = p->sc[b]; p->list[b] = p->list[p->i]; p->sc[b] = p->sc[p->i];
            if (p->pol_on) { p->pol_now = p->pl[b]; p->pl[b] = p->pl[p->i]; }
            p->i++;
            if (m == p->ttm) continue;
            p->bad_now = 0; p->hist = v >= 800000000 ? hist_score(m, p->ply) : (p->pol_on ? hist_score(m, p->ply) : v);
            return m;
        }
        p->stage = ST_BAD; p->i = 0;
        break;
    case ST_BAD:
        while (p->i < p->nbad) {
            int b = p->i;
            for (int j = b + 1; j < p->nbad; j++) if (p->bsc[j] > p->bsc[b]) b = j;
            Move m = p->bad[b]; int sv = p->bsee[b]; p->bad[b] = p->bad[p->i]; p->bsc[b] = p->bsc[p->i]; p->bsee[b] = p->bsee[p->i]; p->i++;
            p->bad_now = 1; p->see_now = sv; p->hist = 0;
            return m;
        }
        p->stage = ST_DONE;
        break;
    default:
        return 0;
    }
}

static int qsearch(int alpha, int beta, int ply) {
    if ((++nodes & check_mask) == 0) check_time();
    if (stop_flag) return 0;
    if (ply >= MAXPLY - 1) return evaluate();
    TTE *e = tt_find(P.key);
    Move ttm = 0;
    if (e->key == P.key) {
        ttm = e->move;
        int s = e->score, fl = e->flag & 3;
        if (s > MATE_IN) s -= ply; else if (s < -MATE_IN) s += ply;
        if (fl == TT_EXACT || (fl == TT_LOWER && s >= beta) || (fl == TT_UPPER && s <= alpha)) return s;
    }
    int chk = in_check(), best, stand = -INF, a0 = alpha;
    if (!chk) {
        corr_ply = ply; stand = best = eval_corr();
        if (best >= beta) return best;
        if (best > alpha) alpha = best;
    } else best = -INF;
    Move bestm = 0, m; int legal = 0;
    thr_stack[ply] = 0;
    if (ttm && !chk && !is_capture(ttm) && !PROMO(ttm)) ttm = 0;       // only captures are searched here
    Picker mp; mp_init(&mp, ttm, ply, !chk);                             // out of check: captures that do not lose material; in check: every move
    while ((m = mp_next(&mp))) {
        if (!chk) {
            int victim = FLAG(m) == F_EP ? PAWN : (P.sq[TO(m)] == EMPTY ? NO_PT : P.sq[TO(m)] % 6);
            if (!PROMO(m) && stand + mat_val[victim] + par[DELTA_Q] <= alpha) continue;   // delta pruning
        }
        move_stack[ply] = m; piece_stack[ply] = P.sq[FROM(m)];
        if (!make(m)) continue;
        legal++;
        int v = -qsearch(-beta, -alpha, ply + 1);
        unmake();
        if (stop_flag) return 0;
        if (v > best) { best = v; bestm = m; if (v > alpha) { alpha = v; if (v >= beta) break; } }
    }
    if (chk && !legal) return -MATE + ply;
    int s = best;
    if (s > MATE_IN) s += ply; else if (s < -MATE_IN) s -= ply;
    e = tt_find(P.key);
    if (e->key != P.key || e->depth == 0) {                             // never overwrite a real search result with a quiescence one
        e->key = P.key; e->move = bestm; e->score = (short)s; e->depth = 0;
        e->flag = TTF(best >= beta ? TT_LOWER : (best > a0 ? TT_EXACT : TT_UPPER));
    }
    return best;
}

static int search(int depth, int alpha, int beta, int ply, int allow_null) {
    int pv_node = beta - alpha > 1;
    if (is_draw()) return 0;
    int chk = in_check();
    if (chk && par[CHK_EXT]) depth++;
    if (depth <= 0) return qsearch(alpha, beta, ply);
    if ((++nodes & check_mask) == 0) check_time();
    if (stop_flag) return 0;
    if (ply >= MAXPLY - 1) return evaluate();
    if (alpha < -MATE + ply) alpha = -MATE + ply;                         // mate-distance pruning
    if (beta > MATE - ply - 1) beta = MATE - ply - 1;
    if (alpha >= beta) return alpha;

    TTE *e = tt_find(P.key);
    Move ttm = 0, excl = excluded[ply];
    int tt_score = 0, tt_depth = -1, tt_flag = TT_NONE, cutnode = cutn[ply];
    if (e->key == P.key) {
        ttm = e->move; tt_depth = e->depth; tt_flag = e->flag & 3;
        tt_score = e->score;
        if (tt_score > MATE_IN) tt_score -= ply; else if (tt_score < -MATE_IN) tt_score += ply;
        if (!pv_node && !excl && tt_depth >= depth &&
            (tt_flag == TT_EXACT || (tt_flag == TT_LOWER && tt_score >= beta) || (tt_flag == TT_UPPER && tt_score <= alpha))) return tt_score;
    }
    thr_stack[ply] = par[THR_HIST] ? attacks_by(P.side ^ 1) : 0;
#ifdef CHECK_THR
    { U64 t = attacks_by(P.side ^ 1); for (int q = 0; q < 64; q++) if (((t >> q) & 1) != (U64)attacked(q, P.side ^ 1)) { printf("attack map disagrees on square %d\n", q); fflush(stdout); abort(); } }
#endif
    corr_ply = ply;
    int ev = chk ? -INF : eval_corr(), raw_ev = ev;
    evals_stack[ply] = ev;
    int improving = !chk && ply >= 2 && ev > evals_stack[ply - 2];
    if (!chk && tt_flag != TT_NONE && tt_score > -MATE_IN && tt_score < MATE_IN &&
        (tt_flag == TT_EXACT || (tt_flag == TT_LOWER && tt_score > ev) || (tt_flag == TT_UPPER && tt_score < ev))) ev = tt_score;
    int vm = 256, vcls = 0;                                                // pruning-margin multiplier (x256) from the reliability read-out; vcls: -1 reliable, +1 unreliable
    if (have_vol && !chk && ((par[VOL_MIX] && !pv_node && depth <= 8) || par[VOL_LMR] || par[VOL_NMP])) {
        int vol = eval_vol();
        if (par[VOL_MIX] && !pv_node && depth <= 8) { vm = 256 + par[VOL_MIX] * (vol * 256 / par[VOL_REF] - 256) / 100; vm = vm < 128 ? 128 : (vm > 512 ? 512 : vm); }
        vcls = vol * 10 <= par[VOL_REF] * 6 ? -1 : (vol * 10 >= par[VOL_REF] * 15 ? 1 : 0);
    }
    if (!pv_node && !chk && !excl && beta > -MATE_IN && beta < MATE_IN) {
        if (depth <= 3 && ev + par[RAZOR] * depth * vm / 256 < alpha) {                     // razoring: hopeless at low depth -> resolve captures and leave
            int v = qsearch(alpha, beta, ply);
            if (v <= alpha) return v;
        }
        if (depth <= 8 && ev - par[RFP] * (depth - improving) * vm / 256 >= beta) return ev;   // reverse futility
        if (allow_null && depth >= 3 && ev >= beta && !(par[NMP_TT] && tt_flag == TT_UPPER && tt_score < beta) &&
            (P.bb[P.side][KNIGHT] | P.bb[P.side][BISHOP] | P.bb[P.side][ROOK] | P.bb[P.side][QUEEN])) {
            int R = par[NMP_BASE] + depth / par[NMP_DIV] + ((ev - beta) / par[NMP_EV] > 3 ? 3 : (ev - beta) / par[NMP_EV]);
            if (par[VOL_NMP]) R -= vcls;                                    // a reliable evaluation: trust the null move more; an unreliable one: less
            move_stack[ply] = 0; cutn[ply + 1] = !cutnode;
            make_null();
            int v = -search(depth - 1 - R, -beta, -beta + 1, ply + 1, 0);
            unmake_null();
            if (stop_flag) return 0;
            if (v >= beta) return v > MATE_IN ? beta : v;
        }
    }
    if (depth >= 4 && !ttm && !excl) depth--;                              // internal iterative reduction
    if (par[CUT_IIR] && cutnode && depth >= 8 && !ttm && !excl) depth--;
    // probcut: a capture that wins enough material and still beats beta by a margin at reduced depth is very likely a cutoff
    if (par[PROBCUT] && !pv_node && !chk && !excl && depth >= 5 && beta > -MATE_IN && beta < MATE_IN) {
        int probbeta = beta + par[PROBCUT] - (improving ? par[PROBCUT] / 3 : 0);
        if (!(tt_flag != TT_NONE && tt_depth >= depth - 3 && tt_score < probbeta)) {
            Move pl[256]; int ps[256];
            int pn = gen(pl, 1), need = probbeta - evals_stack[ply];        // the exchange must win at least this much to be worth trying
            score_moves(pl, ps, pn, 0, ply);
            for (int i = 0; i < pn; i++) {
                Move m = pick(pl, ps, pn, i);
                if (ps[i] < 0) break;
                if (see(m) < need) continue;
                move_stack[ply] = m; piece_stack[ply] = P.sq[FROM(m)];
                if (!make(m)) continue;
                cutn[ply + 1] = !cutnode;
                int v = -qsearch(-probbeta, -probbeta + 1, ply + 1);
                if (v >= probbeta) v = -search(depth - 4, -probbeta, -probbeta + 1, ply + 1, 1);
                unmake();
                if (stop_flag) return 0;
                if (v >= probbeta) {
                    e = tt_find(P.key);
                    if (e->key != P.key || e->depth < depth - 3) { e->key = P.key; e->move = m; e->score = (short)v; e->depth = (unsigned char)(depth - 3); e->flag = TTF(TT_LOWER); }
                    return v;
                }
            }
        }
    }
    Move quiets[256], caps[64];
    int legal = 0, nq = 0, nc = 0, best = -INF, a0 = alpha;
    Move bestm = 0, m;
    Picker mp; mp_init(&mp, ttm, ply, 0);
    mp.pol = have_pol && par[POL_W] && !chk && depth >= par[POL_DEPTH];
    int is_capture_tt = mp.ttm && is_capture(mp.ttm);
    while ((m = mp_next(&mp))) {
        if (m == excl) continue;
        int quiet = !is_capture(m) && !PROMO(m);
        int hs = quiet ? mp.hist : 0;
        int ext = 0;
        // singular extension: if every other move falls clearly short of the table move, search the table move deeper
        if (m == ttm && !excl && depth >= par[SE_DEPTH] && tt_depth >= depth - 3 && (tt_flag == TT_LOWER || tt_flag == TT_EXACT) &&
            tt_score > -MATE_IN && tt_score < MATE_IN && ply + depth < MAXPLY - 8) {
            int sbeta = tt_score - par[SE_MUL] * depth;
            excluded[ply] = m;
            int v = search((depth - 1) / 2, sbeta - 1, sbeta, ply, 0);
            excluded[ply] = 0;
            if (stop_flag) return 0;
            if (v < sbeta) ext = (par[SE_DOUBLE] && !pv_node && v < sbeta - par[SE_DOUBLE] && ply < 2 * root_depth) ? 2 : 1;
            else if (sbeta >= beta) return sbeta;                         // several moves beat beta: cut
            else if (par[NEG_EXT] && (tt_score >= beta || cutnode)) ext = -1;   // not singular and probably a cutoff anyway: look less
        }
        if ((!pv_node || (par[PV_PRUNE] && ply > 0)) && !chk && best > -MATE_IN) {
            if (quiet) {
                // both conditions stay true for every later quiet move of this node, so those are not even generated
                if (depth <= 5 && legal >= (par[LMP_BASE] + depth * depth) / (2 - improving)) { mp.skip_quiets = 1; continue; }   // late-move pruning
                if (depth <= 7 && ev + (par[FUT_BASE] + par[FUT_MUL] * depth) * vm / 256 <= alpha) { mp.skip_quiets = 1; continue; }   // futility
                if (depth <= 3 && hs < -par[HIST_PRUNE] * depth) continue;                               // moves that keep failing
                if (depth <= 6 && see(m) < -par[SEE_Q] * depth) continue;                             // moves into a losing exchange
            } else if (depth <= 6 && mp.bad_now && mp.see_now < -par[SEE_C] * depth) continue;
        }
        move_stack[ply] = m; piece_stack[ply] = P.sq[FROM(m)];
        if (!make(m)) continue;
        legal++;
        int v, nd = depth - 1 + ext;
        if (legal == 1) { cutn[ply + 1] = pv_node ? 0 : !cutnode; v = -search(nd, -beta, -alpha, ply + 1, 1); }
        else {
            int r = 0;
            if (depth >= 3 && (quiet || par[CAP_LMR]) && legal > 1 + 2 * pv_node) {
                r = lmr[depth > 63 ? 63 : depth][legal > 63 ? 63 : legal];
                if (!improving) r++;
                if (cutnode) r += par[CUT_LMR];
                if (par[TTCAP_LMR] && is_capture_tt) r++;
                if (pv_node) r--;
                if (chk || in_check()) r--;
                if (quiet) {
                    if (m == killers[ply][0] || m == killers[ply][1]) r--;
                    if (par[LMR_HIST]) { int h = hs / par[LMR_HIST]; r -= h > 3 ? 3 : (h < -3 ? -3 : h); }     // how well the move has done before, as a sliding scale
                    else r -= hs > 16000 ? 2 : (hs > 6000 ? 1 : (hs < -16000 ? -2 : (hs < -6000 ? -1 : 0)));
                } else r -= mp.bad_now ? 1 : 2;                              // late captures too, but less than quiet moves; winning ones least
                if (par[POL_LMR] && mp.pol_on && quiet && m != mp.ttm) { int dlt = mp.pol_best - mp.pol_now; r += dlt > 1024 ? 1 : (dlt < 128 ? -1 : 0); }   // far below the read-out's favourite: shallower; close to it: deeper
                if (par[VOL_LMR]) r -= vcls;                                 // unreliable evaluation here: look at late moves less shallowly; reliable: more
                if (r < 0) r = 0;
                if (r > nd - 1) r = nd - 1;
            }
            cutn[ply + 1] = r ? 1 : !cutnode;
            v = -search(nd - r, -alpha - 1, -alpha, ply + 1, 1);
            if (v > alpha && r) {
                // the shallow look liked the move. Much more than the best so far: verify one ply deeper; barely: the shallow answer will do
                int nd2 = nd;
                if (par[DEEPER]) nd2 += (v > best + 50 + 2 * nd) - (v < best + 10);
                cutn[ply + 1] = !cutnode;
                if (nd2 > nd - r) v = -search(nd2, -alpha - 1, -alpha, ply + 1, 1);
            }
            if (v > alpha && v < beta) { cutn[ply + 1] = 0; v = -search(nd, -beta, -alpha, ply + 1, 1); }
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
                        int bonus = par[HB_QUAD] * depth * depth + par[HB_LIN] * depth; if (bonus > par[HB_MAX]) bonus = par[HB_MAX];
                        hist_update(m, ply, bonus);
                        for (int q = 0; q < nq; q++) hist_update(quiets[q], ply, -bonus);
                    } else if (is_capture(m) && FLAG(m) != F_EP) {
                        int bonus = par[HB_QUAD] * depth * depth + par[HB_LIN] * depth; if (bonus > par[HB_MAX]) bonus = par[HB_MAX];
                        short *c = &caph[(int)P.sq[FROM(m)]][TO(m)][P.sq[TO(m)] % 6];
                        *c += bonus - *c * bonus / 16384;
                    }
                    if (!quiet || 1) {                                      // captures tried before the cutoff move were not good enough
                        int bonus = par[HB_QUAD] * depth * depth + par[HB_LIN] * depth; if (bonus > par[HB_MAX]) bonus = par[HB_MAX];
                        for (int q = 0; q < nc; q++) { short *c = &caph[(int)P.sq[FROM(caps[q])]][TO(caps[q])][P.sq[TO(caps[q])] % 6]; *c -= bonus + *c * bonus / 16384; }
                    }
                    break;
                }
            }
        }
        if (quiet && nq < 255) quiets[nq++] = m;
        else if (!quiet && nc < 63 && is_capture(m) && FLAG(m) != F_EP) caps[nc++] = m;
    }
    if (!legal) return excl ? alpha : (chk ? -MATE + ply : 0);
    if (excl) return best;
    if (!chk && best > -MATE_IN && best < MATE_IN && (!bestm || (!is_capture(bestm) && !PROMO(bestm))) &&
        !(best >= beta && best <= raw_ev) && !(best <= a0 && best >= raw_ev)) {       // learn how far off the static evaluation was
        int *c = &corrhist[P.side][P.pkey & 16383], w = depth + 1 > 16 ? 16 : depth + 1, d = (best - raw_ev) * 256;
        d = d > 32768 ? 32768 : (d < -32768 ? -32768 : d);
        *c = (*c * (256 - w) + d * w) / 256;
        *c = *c > 12288 ? 12288 : (*c < -12288 ? -12288 : *c);
        for (int k = 0; k < 2; k++) {
            c = &corr_np[P.side][k][P.npkey[k] & 16383];
            *c = (*c * (256 - w) + d * w) / 256;
            *c = *c > 12288 ? 12288 : (*c < -12288 ? -12288 : *c);
        }
        if (ply > 0 && move_stack[ply - 1]) {
            c = &corr_cont[P.side][piece_stack[ply - 1]][TO(move_stack[ply - 1])];
            *c = (*c * (256 - w) + d * w) / 256;
            *c = *c > 12288 ? 12288 : (*c < -12288 ? -12288 : *c);
        }
        if (par[LEARN_MIX] && have_net) learn_update(d / 256, w);
    }
    int s = best;
    if (s > MATE_IN) s += ply; else if (s < -MATE_IN) s -= ply;
    e = tt_find(P.key);
    e->key = P.key; e->move = bestm; e->score = (short)s; e->depth = (unsigned char)depth;
    e->flag = TTF(best >= beta ? TT_LOWER : (best > a0 ? TT_EXACT : TT_UPPER));
    if (harvest_on && best > a0 && best < beta && depth >= 4 && ply >= 2 && !chk && bestm && !is_capture(bestm) && !PROMO(bestm) &&
        best > -2000 && best < 2000) {
        Rec *r = &harvest[n_harvest < 256 ? n_harvest : (int)(rnd64() % 256)]; fill_rec(r, best); r->pad = (signed char)-(depth > 100 ? 100 : depth);
        if (n_harvest < 256) n_harvest++;
    }
    return best;
}

static void init_lmr(void) { for (int d = 1; d < 64; d++) for (int i = 1; i < 64; i++) lmr[d][i] = (int)(par[LMR_BASE] / 100.0 + log(d) * log(i) / (par[LMR_DIV] / 100.0)); }
static void move_str(Move m, char *o) {
    int f = FROM(m), t = TO(m);
    o[0] = 'a' + (f & 7); o[1] = '1' + (f >> 3); o[2] = 'a' + (t & 7); o[3] = '1' + (t >> 3); o[4] = 0;
    if (PROMO(m)) { o[4] = "?nbrq"[PROMO(m)]; o[5] = 0; }
}

// root: iterative deepening, PVS over the root moves, optional additive prior (root_bonus) from the judgment network
static Move root_moves[256]; static int root_scores[256], n_root, use_bonus, print_root, silent, root_score;
static Move search_root(int max_depth) {
    Move list[256];
    int n = gen(list, 0);
    n_root = 0;
    for (int i = 0; i < n; i++) if (make(list[i])) { unmake(); root_moves[n_root++] = list[i]; }
    if (!n_root) return 0;
    Move best = root_moves[0];
    memset(killers, 0, sizeof killers);
    corr_ply = 0; evals_stack[0] = in_check() ? -INF : eval_corr();
    root_vol = (have_vol && !in_check()) ? eval_vol() : 0;
    for (int i = 0; i < 2; i++) for (int t = 0; t < 4; t++) for (int a = 0; a < 64; a++) for (int b = 0; b < 64; b++) history_t[i][t][a][b] /= 2;
    thr_stack[0] = 0;
    int last = 0, stable = 0;
    tt_age = (tt_age + 1) & 63;
    memset(root_move_nodes, 0, sizeof root_move_nodes);
    for (root_depth = 1; root_depth <= max_depth; root_depth++) {
        int alpha = -INF, beta = INF, iter_best = -INF; Move iter_move = 0;
        int delta = par[ASP], first_done = 0;
        if (root_depth >= 5) { alpha = last - delta; beta = last + delta; }
        for (;;) {
            int a = alpha; iter_best = -INF; iter_move = 0; first_done = 0;
            for (int i = 0; i < n_root; i++) {
                Move m = root_moves[i];
                int bon = use_bonus ? root_bonus[FROM(m) * 64 + TO(m)] : 0;
                move_stack[0] = m; piece_stack[0] = P.sq[FROM(m)];
                U64 n0 = nodes; int rq = !is_capture(m) && !PROMO(m);
                make(m);
                int v;
                if (i == 0) { cutn[1] = 0; v = -search(root_depth - 1, -(beta - bon), -(a - bon), 1, 1) + bon; }
                else {
                    // late quiet root moves get a shallower first look, like late moves everywhere else in the tree
                    int r = 0;
                    if (par[ROOT_LMR] && rq && i >= 3 && root_depth >= 4 && !in_check()) {
                        r = lmr[root_depth > 63 ? 63 : root_depth][i > 63 ? 63 : i] - 1;
                        if (r < 0) r = 0;
                        if (r > root_depth - 2) r = root_depth - 2;
                    }
                    cutn[1] = 1; v = -search(root_depth - 1 - r, -(a - bon) - 1, -(a - bon), 1, 1) + bon;
                    if (v > a && r && !stop_flag) v = -search(root_depth - 1, -(a - bon) - 1, -(a - bon), 1, 1) + bon;
                    if (v > a && !stop_flag) { cutn[1] = 0; v = -search(root_depth - 1, -(beta - bon), -(a - bon), 1, 1) + bon; }
                }
                unmake();
                root_move_nodes[FROM(m) * 64 + TO(m)] += nodes - n0;
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
        int prev_score = last; stable = (iter_move == best && root_depth > 1) ? stable + 1 : 0;
        best = iter_move; last = iter_best; root_score = last;
        for (int i = 1; i < n_root; i++) if (root_moves[i] == best) { memmove(root_moves + 1, root_moves, i * sizeof(Move)); root_moves[0] = best; break; }
        if (par[ROOT_ORD])                                                 // the rest in the order of how much search each has needed so far: refutations first
            for (int i = 2; i < n_root; i++) {
                Move mv = root_moves[i]; U64 k = root_move_nodes[FROM(mv) * 64 + TO(mv)]; int j = i - 1;
                while (j >= 1 && root_move_nodes[FROM(root_moves[j]) * 64 + TO(root_moves[j])] < k) { root_moves[j + 1] = root_moves[j]; j--; }
                root_moves[j + 1] = mv;
            }
        char ms[8]; move_str(best, ms);
        int t = elapsed_ms();
        if (silent) { if (n_root == 1 && root_depth >= 2) break; if (last > MATE_IN && root_depth >= 6) break; continue; }
        if (last > MATE_IN) printf("info depth %d score mate %d nodes %llu time %d pv %s\n", root_depth, (MATE - last + 1) / 2, nodes, t, ms);
        else if (last < -MATE_IN) printf("info depth %d score mate %d nodes %llu time %d pv %s\n", root_depth, -(MATE + last + 1) / 2, nodes, t, ms);
        else printf("info depth %d score cp %d nodes %llu time %d pv %s\n", root_depth, last, nodes, t, ms);
        fflush(stdout);
        if (time_ms_limit && tm_dynamic && root_depth >= 6) {
            // Spend less when the choice is settled, more when it is not. Three signs, multiplied:
            //  - the share of all searched positions that went into the chosen move (a large share = nothing else was worth looking at)
            //  - for how many depths in a row the choice has stayed the same
            //  - whether the score just fell (a problem was found: look for the way out)
            double frac = nodes ? (double)root_move_nodes[FROM(best) * 64 + TO(best)] / (double)nodes : 1.0;
            double scale = (1.5 - frac) * 1.35 * (1.3 - 0.05 * (stable > 10 ? 10 : stable));
            int drop = prev_score - last; if (drop > 0) scale *= 1.0 + (drop > 75 ? 75 : drop) * 0.004;
            if (par[VOL_TM] && root_vol) { double f = (double)root_vol / par[VOL_REF]; f = f < 0.5 ? 0.5 : (f > 2.0 ? 2.0 : f); scale *= 0.7 + 0.3 * f; }   // an unreliable evaluation at the root: think longer
            if (t >= (int)(time_ms_limit * scale)) break;
        } else
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
    acc_refresh();
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

// ------------------------------------------------------------------ self-play: the engine's only teacher is its own games
// record for OURO (the transformer): the whole position in its own encoding (side to move = "own", moving up the board),
// the move the search chose, its score and (filled in at the end of the game) the result. 72 bytes.
typedef struct { unsigned char board[64]; unsigned char castle, ep; unsigned short move; short score; signed char result; unsigned char ply; } ORec;
static void fill_orec(ORec *r, Move m, int sc, int ply) {
    int us = P.side;
    memset(r->board, 0, 64);
    for (int c = 0; c < 2; c++) for (int pt = 0; pt < 6; pt++) {
        U64 b = P.bb[c][pt];
        while (b) { int s = lsb(b); b &= b - 1; r->board[us == WHITE ? s : s ^ 56] = (unsigned char)((c == us ? 1 : 7) + pt); }
    }
    r->castle = (unsigned char)(us == WHITE ? P.castle : (((P.castle >> 2) & 3) | ((P.castle & 3) << 2)));
    r->ep = (unsigned char)(P.ep == NOSQ ? 64 : (us == WHITE ? P.ep : P.ep ^ 56));
    int f = FROM(m), t = TO(m), pr = PROMO(m);
    if (us == BLACK) { f ^= 56; t ^= 56; }
    r->move = (unsigned short)((pr && pr != QUEEN) ? 4096 + ((f & 7) * 8 + (t & 7)) * 3 + (pr - 1) : f * 64 + t);
    r->score = (short)(sc > 20000 ? 20000 : (sc < -20000 ? -20000 : sc)); r->result = 0; r->ply = (unsigned char)(ply > 255 ? 255 : ply);
}
static int legal_list(Move *out) {
    Move l[256]; int n = gen(l, 0), k = 0;
    for (int i = 0; i < n; i++) if (make(l[i])) { unmake(); out[k++] = l[i]; }
    return k;
}
static void selfplay(const char *path, int ngames, int nodes_per_move, U64 seed, int adjudicate, int tree_n, int ouro) {
    FILE *f = fopen(path, "ab");
    if (!f) { printf("cannot open %s\n", path); return; }
    FILE *fo = 0; static ORec orecs[1024]; static signed char oside[1024];
    if (ouro) { char op[1024]; snprintf(op, sizeof op, "%s.o", path); fo = fopen(op, "ab"); }
    rng_state ^= seed * 0x9E3779B97F4A7C15ULL + 12345;
    static Rec recs[1024]; static signed char rside[1024];
    silent = 1; tt_alloc(8); harvest_on = tree_n > 0; check_mask = 255;
    long long tot_pos = 0, tot_ply = 0, tot_tree = 0; int res_cnt[3] = {0, 0, 0};
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    for (int g = 0; g < ngames; g++) {
        set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        memset(tt, 0, (tt_mask + 1) * sizeof(TTE)); memset(history_t, 0, sizeof history_t); memset(cmh, 0, sizeof cmh);
        memset(fuh, 0, sizeof fuh); memset(caph, 0, sizeof caph); memset(corrhist, 0, sizeof corrhist); memset(FW, 0, sizeof FW); memset(FWq, 0, sizeof FWq); memset(corr_np, 0, sizeof corr_np); memset(corr_cont, 0, sizeof corr_cont);
        Move ml[256]; int ok = 1, nrand = 8 + (int)(rnd64() % 2);
        for (int i = 0; i < nrand; i++) { int n = legal_list(ml); if (!n) { ok = 0; break; } make(ml[rnd64() % n]); }
        if (!ok) { g--; continue; }
        int nrec = 0, norec = 0, result = 0, winc = 0, drawc = 0, first = 1;
        for (int ply = 0;; ply++) {
            int n = legal_list(ml);
            if (!n) { result = in_check() ? (P.side == WHITE ? -1 : 1) : 0; break; }
            if (is_draw() || ply >= 400) { result = 0; break; }
            nodes = 0; stop_flag = 0; node_limit = nodes_per_move; hard_ms_limit = time_ms_limit = 0; use_bonus = 0; n_harvest = 0;
            Move m = search_root(64);
            int sc = root_score;
            if (!m) m = ml[0];
            if (first && adjudicate && abs(sc) > 500) { ok = 0; break; }          // the random opening already decided the game: discard it
            first = 0;
            if (fo && norec < 1024) { fill_orec(&orecs[norec], m, sc, ply); oside[norec++] = (signed char)P.side; }
            if (!in_check() && !is_capture(m) && !PROMO(m) && abs(sc) < MATE_IN && nrec < 1024) {
                fill_rec(&recs[nrec], sc); recs[nrec].pad = (signed char)(ply > 127 ? 127 : ply);
                rside[nrec++] = (signed char)P.side;
            }
            for (int t = 0; t < tree_n && n_harvest > 0; t++) {              // a few of the positions this search resolved exactly
                int j = (int)(rnd64() % n_harvest);
                harvest[j].result = (signed char)(t >= 4);                  // marks the fifth and later tree positions of a move (for experiments)
                fwrite(&harvest[j], sizeof(Rec), 1, f); tot_tree++;
                harvest[j] = harvest[--n_harvest];
            }
            if (adjudicate) {
                if (abs(sc) >= 1500) { if (++winc >= 4) { result = ((sc > 0) == (P.side == WHITE)) ? 1 : -1; break; } } else winc = 0;
                if (ply >= 80 && abs(sc) <= 10) { if (++drawc >= 16) { result = 0; break; } } else drawc = 0;
            }
            make(m); tot_ply++;
        }
        if (!ok) { g--; continue; }
        for (int i = 0; i < nrec; i++) recs[i].result = (signed char)(rside[i] == WHITE ? result : -result);
        fwrite(recs, sizeof(Rec), nrec, f);
        if (fo) { for (int i = 0; i < norec; i++) orecs[i].result = (signed char)(oside[i] == WHITE ? result : -result); fwrite(orecs, sizeof(ORec), norec, fo); }
        tot_pos += nrec; res_cnt[result + 1]++;
        if ((g + 1) % 500 == 0) { fflush(f); printf("games %d positions %lld plies/game %.0f  white wins %d draws %d black wins %d  %.1f games/s\n", g + 1, tot_pos, (double)tot_ply / (g + 1), res_cnt[2], res_cnt[1], res_cnt[0], (g + 1) * 1000.0 / (elapsed_ms() + 1)); fflush(stdout); }
    }
    fclose(f); if (fo) fclose(fo);
    printf("done games %d positions %lld tree positions %lld white %d draw %d black %d\n", ngames, tot_pos, tot_tree, res_cnt[2], res_cnt[1], res_cnt[0]);
}

int main(int argc, char **argv) {
    init_tables(); init_pext(); init_zobrist(); tt_alloc(32);
    accs = (Acc *)big_alloc(sizeof(Acc) * 2048);
    init_lmr();
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    if (argc > 1 && !strcmp(argv[1], "mob")) { eval_mode = 1; int unit[5] = {100, 100, 100, 100, 100}; set_piece_values(unit); }
    else if (argc > 1) load_net(argv[1]);
    if (argc > 6 && !strcmp(argv[2], "selfplay")) {       // calc <weights|mob> selfplay <out> <games> <nodes> <seed> [adjudicate] [tree positions per move] [1 = also write <out>.o for the transformer]
        selfplay(argv[3], atoi(argv[4]), atoi(argv[5]), strtoull(argv[6], 0, 10), argc > 7 ? atoi(argv[7]) : 1, argc > 8 ? atoi(argv[8]) : 0, argc > 9 ? atoi(argv[9]) : 0);
        return 0;
    }
    static char line[65536];
    setvbuf(stdout, NULL, _IOLBF, 0);
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!strcmp(line, "uci")) {
            printf("id name CALC\nid author loopchess\noption name EvalFile type string default <none>\noption name Hash type spin default 32 min 1 max 4096\noption name Bonus type string default <none>\n");
            for (int i = 0; i < NPAR; i++) printf("option name %s type spin default %d min 0 max 100000\n", par_name[i], par[i]);
            printf("uciok\n");
        } else if (!strcmp(line, "isready")) printf("readyok\n");
        else if (!strncmp(line, "setoption name EvalFile value ", 30)) { if (!load_net(line + 30)) printf("info string cannot load %s\n", line + 30); }
        else if (!strncmp(line, "setoption name Hash value ", 26)) tt_alloc(atoi(line + 26));
        else if (!strncmp(line, "setoption name TmNew value ", 27)) tm_new = atoi(line + 27);
        else if (!strncmp(line, "setoption name TmSoft value ", 28)) tm_soft = atoi(line + 28);
        else if (!strncmp(line, "setoption name ", 15) && strstr(line, " value ") && strncmp(line + 15, "Bonus", 5) && strncmp(line + 15, "EvalFile", 8)) {
            char *v = strstr(line, " value "); int len = (int)(v - (line + 15));
            for (int i = 0; i < NPAR; i++) if ((int)strlen(par_name[i]) == len && !strncmp(line + 15, par_name[i], len)) { par[i] = atoi(v + 7); if (par[i] < 1 && i != LMP_BASE && i < PROBCUT) par[i] = 1; }
            init_lmr();
        }
        else if (!strcmp(line, "ucinewgame")) { memset(tt, 0, (tt_mask + 1) * sizeof(TTE)); memset(history_t, 0, sizeof history_t); memset(cmh, 0, sizeof cmh); memset(fuh, 0, sizeof fuh); memset(caph, 0, sizeof caph); memset(corrhist, 0, sizeof corrhist); memset(FW, 0, sizeof FW); memset(FWq, 0, sizeof FWq); memset(corr_np, 0, sizeof corr_np); memset(corr_cont, 0, sizeof corr_cont); }
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
            time_ms_limit = hard_ms_limit = 0; tm_dynamic = 0;
            if (movetime) { hard_ms_limit = movetime > 12 ? movetime - 6 : movetime; time_ms_limit = hard_ms_limit; }
            else if (wtime >= 0 || btime >= 0) {
                int t = P.side == WHITE ? wtime : btime, inc = P.side == WHITE ? winc : binc;
                int alloc = t / 30 + inc * 3 / 4; if (alloc > t / 3) alloc = t / 3; if (alloc < 5) alloc = 5;
                time_ms_limit = alloc * 6 / 10; hard_ms_limit = alloc * 2; if (hard_ms_limit > t / 2) hard_ms_limit = t / 2;
                if (tm_new) {                                             // a larger share per move as the base, a wider reach for hard positions
                    alloc = t / 20 + inc * 3 / 4; if (alloc > t / 3) alloc = t / 3; if (alloc < 5) alloc = 5;
                    time_ms_limit = alloc * tm_soft / 100; hard_ms_limit = alloc * 3; if (hard_ms_limit > t * 2 / 5) hard_ms_limit = t * 2 / 5;
                    tm_dynamic = 1;
                }
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
        else if (!strcmp(line, "pol")) {                 // the read-out's scores for the quiet moves of the current position
            if (have_pol && !in_check()) { evaluate(); Picker q; mp_init(&q, 0, 0, 0); q.pol = 1; Move m; char ms[8]; while ((m = mp_next(&q))) if (q.pol_on && !is_capture(m) && !PROMO(m)) { move_str(m, ms); printf("pol %s %d\n", ms, q.pol_now); } }
            printf("pol end\n");
        }
        else if (!strcmp(line, "vol")) { evaluate(); printf("vol %d\n", have_vol ? eval_vol() : -1); }
        else if (!strncmp(line, "see ", 4)) { Move m = parse_move(line + 4); printf("see %d\n", m ? see(m) : -99999); }
        else if (!strcmp(line, "quit")) break;
        fflush(stdout);
    }
    return 0;
}
