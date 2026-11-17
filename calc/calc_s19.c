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
typedef struct { U64 bb[2][6], occ[2]; signed char sq[64]; int side, castle, ep, half; U64 key, pkey, npkey[2], mmkey[2]; } Pos;      // mmkey: where the minor pieces and kings stand / the major pieces and kings
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
typedef struct { short a[2][HMAX] __attribute__((aligned(32))); float lin[2][2]; int pb, ok[2], need[2], nchg, ksq[2]; int chg[6][4]; } Acc;   // sums as 16-bit integers (see quantize_net); ok: one flag per view; need: this move took the view's king to another table
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
// Each hidden unit has its own integer scale 2^-qu: the finest at which the largest sum that unit could ever produce still
// fits 16 bits (QU_ONE = 1.0 in the unit's scale, QU_MUL brings a clipped sum to 15 bits whatever the scale).
static short QU_ONE[HMAX + 16] __attribute__((aligned(32))), QU_MUL[HMAX + 16] __attribute__((aligned(32))); static unsigned char QU[HMAX];
// reliability read-out (optional trailer 'VOL1'): predicts how far a search will move the static score of this position
// "where to search" read-out (optional trailer 'POL1', zero/pol_train.py): from the hidden layer, a score for every quiet move
#define PHMAX 64
static float *PW1, PB1[PHMAX], PW2[448][PHMAX], PB2[448]; static int NPH = 0, have_pol = 0;
static float NV[NBMAX][2 * HMAX + 1]; static short NVOq[NBMAX][HMAX], NVPq[NBMAX][HMAX]; static int have_vol = 0;
// layered read-out (optional trailer 'MLP1', trainer option --head): on top of the direct read-out, the hidden layer is read
// by MH1 squared clipped units, those by MH2 clipped units, and those are summed -- one such stack per piece-count bucket.
// The first layer is the expensive one (2*NH inputs for each of MH1 units): 16-bit integers, weights in units of 2^-10.
// Four hidden units in five are exactly zero in a real position, so the layer is read SPARSELY: inputs are taken two at
// a time, pairs that are both zero are skipped, and a pair that is not adds its two weight columns to all MH1 sums with
// one multiply-add per eight sums. Layout: [bucket][pair of inputs][unit][the pair's two weights].
#define M1MAX 32
#define M2MAX 64
static int have_mlp = 0, MH1 = 0, MH2 = 0, MHK = 0;      // MHK: the layers read the first MHK units of each side (trailer 'MLP2'; all of them with 'MLP1')
static short *ML1q;
// When the layers read few units (MHK <= 128 a side) everything they need for one bucket lies in two small blocks that are
// read front to back and fetched ahead at the start of the evaluation: MLD = first-layer weights [16 inputs][unit][16],
// MLF = the float layers (bias 1, second layer one input after the other, bias 2, last layer). Scattered tables made
// this small computation wait on cache misses: the 16 -> 32 layer alone took 140 cycles for 40 cycles of arithmetic.
static short *MLD; static float *MLF; static int mlf_stride = 0, mlp_dense = 0;
static float MB1[NBMAX][M1MAX], ML2[NBMAX][M1MAX][M2MAX] __attribute__((aligned(32))), MB2[NBMAX][M2MAX] __attribute__((aligned(32))), ML3[NBMAX][M2MAX] __attribute__((aligned(32)));   // ML2[bucket][input][output]: one input's weights lie together
static int have_net = 0, taper = 0;   // taper: second (endgame) linear table, blended by remaining material
// Mirrored inputs (optional trailer 'MIR1', trainer option --hm): each view's hidden-layer inputs are flipped left-right
// when that view's own king stands on files a-d, so the king is always on files e-h and one set of weights serves both
// wings. Costs nothing per position; the sums are rebuilt when a king crosses the middle (rare).
static int NMIR = 0;
static int NKB = 1;                      // king buckets for the hidden-layer inputs (1 = none); mirrored so the king is on files a-d
static const int KBT7[32] = {0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
// 16 mirrored buckets: every king square of the first two ranks its own, then half-ranks (the last three ranks together)
static const int KBT16[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 14, 14, 15, 15, 14, 14, 15, 15};
static const int KBT32[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};   // every mirrored king square its own table
static const int *KBT = KBT7;
// 11 un-mirrored buckets by own-king square (own side moving up): back rank in six groups, second rank in three, then two bands
static const int KB11[64] = {0, 0, 1, 2, 3, 4, 5, 5,  6, 6, 6, 7, 7, 8, 8, 8,  9, 9, 9, 9, 9, 9, 9, 9,  9, 9, 9, 9, 9, 9, 9, 9,
                             10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
static inline int feat_ix(int v, int c, int pt, int s) { return ((c == v ? 0 : 6) + pt) * 64 + (v == WHITE ? s : s ^ 56); }
static inline int feat_fx(int pb, int v, int c, int pt, int s, int ksq) {       // row of the input matrix (depends on the bucket scheme)
    int ix = feat_ix(v, c, pt, s);
    if (NKB == 1) {
        if (!NMIR) return ix;
        int kv1 = v == WHITE ? ksq : ksq ^ 56;
        return (kv1 & 7) < 4 ? (ix ^ 7) : ix;
    }
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
    else if (NMIR) { int kv = v == WHITE ? ksq : ksq ^ 56; if ((kv & 7) < 4) fx = ix ^ 7; }
    const short *e = NEq + (size_t)fx * NH; short *a = A->a[v];
    if (sign > 0) { for (int i = 0; i < NH; i++) a[i] += e[i]; A->lin[v][0] += NS[ix]; A->lin[v][1] += NS2[ix]; }
    else { for (int i = 0; i < NH; i++) a[i] -= e[i]; A->lin[v][0] -= NS[ix]; A->lin[v][1] -= NS2[ix]; }
}
static inline void put(int c, int pt, int s) { P.bb[c][pt] |= BIT(s); P.occ[c] |= BIT(s); P.sq[s] = c * 6 + pt; P.key ^= zpiece[c * 6 + pt][s]; if (pt == PAWN) P.pkey ^= zpiece[c * 6][s]; else { P.npkey[c] ^= zpiece[c * 6 + pt][s]; if (pt != ROOK && pt != QUEEN) P.mmkey[0] ^= zpiece[c * 6 + pt][s]; if (pt != KNIGHT && pt != BISHOP) P.mmkey[1] ^= zpiece[c * 6 + pt][s]; } }
static inline void rem(int c, int pt, int s) { P.bb[c][pt] ^= BIT(s); P.occ[c] ^= BIT(s); P.sq[s] = EMPTY; P.key ^= zpiece[c * 6 + pt][s]; if (pt == PAWN) P.pkey ^= zpiece[c * 6][s]; else { P.npkey[c] ^= zpiece[c * 6 + pt][s]; if (pt != ROOK && pt != QUEEN) P.mmkey[0] ^= zpiece[c * 6 + pt][s]; if (pt != KNIGHT && pt != BISHOP) P.mmkey[1] ^= zpiece[c * 6 + pt][s]; } }

static inline int phase_bucket(void) { int b = (popc(P.occ[0] | P.occ[1]) - 2) / 4; return b > 7 ? 7 : b; }
// King buckets (NKB 7, mirrored; NKB 11; or mirrored inputs alone): a view's rows of the input matrix depend on where its own
// king stands. When a king moves to another bucket or across the middle, only that view's sums change wholesale. They are
// rebuilt from a small cache: for every view and bucket, the sums of the last position seen there and its piece sets. The
// rebuild adds and removes only the pieces that differ (a handful along a search path) instead of all of them.
typedef struct { short a[HMAX] __attribute__((aligned(32))); U64 bb[2][6]; } KCache;
static KCache *kcache;                   // [view * 64 + table]
static int use_kc = 0;
#ifdef KSTAT
static U64 ks_ens, ks_build, ks_ref, ks_diff;            // evaluations, lazy view builds, view rebuilds from the cache, pieces added or removed there
#endif
static inline int kb_key(int v, int ksq) {              // which table of the input matrix the view reads: bucket and wing
    int kv = v == WHITE ? ksq : ksq ^ 56;
    if (NKB == 11) return KB11[kv];
    int mir = (kv & 7) >= 4;
    if (NKB == 1) return (kv & 7) < 4;
    int kq = kv ^ (mir ? 7 : 0);
    return KBT[(kq >> 3) * 4 + (kq & 7)] * 2 + mir;
}
static void kc_reset(void) {
    if (!kcache) kcache = (KCache *)aligned_alloc(64, (sizeof(KCache) * 128 + 63) / 64 * 64);
    for (int k = 0; k < 128; k++) { memcpy(kcache[k].a, NBq, sizeof(short) * HMAX); memset(kcache[k].bb, 0, sizeof kcache[k].bb); }
    use_kc = have_net && NH && ((NKB > 1 && NKB != 8) || NMIR);
}
static void acc_refresh_view(int v) {
    Acc *A = &accs[hply];
    A->pb = phase_bucket();
    int ksq = lsb(P.bb[v][KING]); A->ksq[v] = ksq;
    if (use_kc) {
        KCache *K = &kcache[v * 64 + kb_key(v, ksq)]; short *a = K->a; float l0 = 0, l1 = 0; int n = NH;
        for (int c = 0; c < 2; c++) for (int pt = 0; pt < 6; pt++) {
            U64 cur = P.bb[c][pt], old = K->bb[c][pt], add = cur & ~old, del = old & ~cur; K->bb[c][pt] = cur;
#ifdef KSTAT
            ks_diff += popc(add) + popc(del);
#endif
            for (; add; add &= add - 1) { const short *e = NEq + (size_t)feat_fx(0, v, c, pt, lsb(add), ksq) * n; for (int i = 0; i < n; i++) a[i] += e[i]; }
            for (; del; del &= del - 1) { const short *e = NEq + (size_t)feat_fx(0, v, c, pt, lsb(del), ksq) * n; for (int i = 0; i < n; i++) a[i] -= e[i]; }
            for (; cur; cur &= cur - 1) { int ix = feat_ix(v, c, pt, lsb(cur)); l0 += NS[ix]; l1 += NS2[ix]; }
        }
        memcpy(A->a[v], a, sizeof(short) * n); A->lin[v][0] = l0; A->lin[v][1] = l1;
#ifdef KSTAT
        ks_ref++;
#endif
        return;
    }
    memcpy(A->a[v], NBq, sizeof(short) * NH); A->lin[v][0] = A->lin[v][1] = 0;
    for (int c = 0; c < 2; c++) for (int pt = 0; pt < 6; pt++) { U64 b = P.bb[c][pt]; while (b) { int s = lsb(b); b &= b - 1; acc_apply(A, v, 1, c, pt, s, ksq); } }
}
static void acc_refresh(void) { if (have_net) { acc_refresh_view(WHITE); acc_refresh_view(BLACK); accs[hply].ok[0] = accs[hply].ok[1] = 1; accs[hply].need[0] = accs[hply].need[1] = 0; } }
// board changes of the move being made
static int chg_n, chg[6][4];
static inline void putA(int c, int pt, int s) { put(c, pt, s); chg[chg_n][0] = 1; chg[chg_n][1] = c; chg[chg_n][2] = pt; chg[chg_n][3] = s; chg_n++; }
static inline void remA(int c, int pt, int s) { rem(c, pt, s); chg[chg_n][0] = -1; chg[chg_n][1] = c; chg[chg_n][2] = pt; chg[chg_n][3] = s; chg_n++; }
// make() calls this: remember the change, compute nothing. A move that takes a king to another table of the input matrix
// only marks that view; the view is rebuilt from the board when (and where) a position below is actually evaluated, so
// king moves that turn out illegal or are cut off by the hash table cost nothing.
static inline void acc_record(int king_moved) {
    if (!have_net) return;
    Acc *A = &accs[hply]; const Acc *S = &accs[hply - 1];
    if (NKB == 8 && phase_bucket() != S->pb) { acc_refresh(); return; }
    A->ok[0] = A->ok[1] = 0; A->nchg = chg_n; A->pb = S->pb;
    A->ksq[0] = lsb(P.bb[0][KING]); A->ksq[1] = lsb(P.bb[1][KING]);
    for (int i = 0; i < chg_n; i++) { A->chg[i][0] = chg[i][0]; A->chg[i][1] = chg[i][1]; A->chg[i][2] = chg[i][2]; A->chg[i][3] = chg[i][3]; }
    A->need[0] = A->need[1] = 0;
    if (king_moved && use_kc)
        for (int v = 0; v < 2; v++) A->need[v] = A->ksq[v] != S->ksq[v] && kb_key(v, A->ksq[v]) != kb_key(v, S->ksq[v]);
}
// build one view of accs[j] from the computed accs[j-1] in ONE pass over the data (no copy first)
static void acc_build(int j, int v) {
    Acc *A = &accs[j]; const Acc *S = &accs[j - 1];
    int n = NH, nc = A->nchg;
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
    A->ok[v] = 1;
}
#ifdef ACC_CHECK
static void acc_check(void) {            // the lazily built sums against a rebuild from the board (test builds only)
    const Acc *A = &accs[hply];
    for (int v = 0; v < 2; v++) {
        short a[HMAX]; int ksq = lsb(P.bb[v][KING]); float l0 = 0, l1 = 0; memcpy(a, NBq, sizeof(short) * NH);
        for (int c = 0; c < 2; c++) for (int pt = 0; pt < 6; pt++) for (U64 b = P.bb[c][pt]; b; b &= b - 1) {
            const short *e = NEq + (size_t)feat_fx(A->pb, v, c, pt, lsb(b), ksq) * NH; for (int i = 0; i < NH; i++) a[i] += e[i];
            int ix = feat_ix(v, c, pt, lsb(b)); l0 += NS[ix]; l1 += NS2[ix];
        }
        if (memcmp(a, A->a[v], sizeof(short) * NH) || fabsf(l0 - A->lin[v][0]) > 0.01f || fabsf(l1 - A->lin[v][1]) > 0.01f || A->ksq[v] != ksq) { printf("info string ACCUMULATOR MISMATCH view %d ply %d\n", v, hply); fflush(stdout); abort(); }
    }
}
#endif
static inline void acc_ensure(void) {
    for (int v = 0; v < 2; v++) {
        int j = hply, rebuild = 0;
        while (!accs[j].ok[v]) { if (accs[j].need[v]) { rebuild = 1; break; } j--; }
        if (rebuild) { int pb = accs[hply].pb; acc_refresh_view(v); accs[hply].pb = pb; accs[hply].ok[v] = 1; continue; }   // the king changed table on the way here: from the board, via the bucket cache
        for (j++; j <= hply; j++) { acc_build(j, v);
#ifdef KSTAT
            ks_build++;
#endif
        }
    }
#ifdef KSTAT
    ks_ens++;
#endif
#ifdef ACC_CHECK
    acc_check();
#endif
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
// Does a cheaper enemy piece attack one of the mover's pieces (pawn on a piece, minor on rook or queen, rook on queen)?
// A static evaluation cannot be trusted to stand in such a position (options ThrRfp, ThrLmr).
static int cheap_threat(void) {
    int us = P.side, th = us ^ 1; U64 occ = P.occ[0] | P.occ[1], b = P.bb[th][PAWN], a;
    const U64 FA = 0x0101010101010101ULL, FH = 0x8080808080808080ULL;
    a = th == WHITE ? (((b & ~FA) << 7) | ((b & ~FH) << 9)) : (((b & ~FA) >> 9) | ((b & ~FH) >> 7));
    if (a & (P.bb[us][KNIGHT] | P.bb[us][BISHOP] | P.bb[us][ROOK] | P.bb[us][QUEEN])) return 1;
    U64 heavy = P.bb[us][ROOK] | P.bb[us][QUEEN];
    if (!heavy) return 0;
    a = 0;
    for (b = P.bb[th][KNIGHT]; b; b &= b - 1) a |= knight_att[lsb(b)];
    for (b = P.bb[th][BISHOP]; b; b &= b - 1) a |= bishop_att(lsb(b), occ);
    if (a & heavy) return 1;
    if (!P.bb[us][QUEEN]) return 0;
    a = 0;
    for (b = P.bb[th][ROOK]; b; b &= b - 1) a |= rook_att(lsb(b), occ);
    return (a & P.bb[us][QUEEN]) != 0;
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
    // (A view reads one king bucket's table at a time, so the bound is taken bucket by bucket.)
    // Per unit: the largest sum above zero and the largest below, each the bias plus the 32 largest per-square maxima of
    // that sign (a square holds one piece, a board 32 pieces), bucket by bucket. Sums wrap consistently in 16 bits, so only
    // the final value has to fit; a single scale for all units (the old rule) was set by the worst unit's negative side.
    float worst = 0; int hist_q[16] = {0};
    for (int i = 0; i < NH; i++) {
        float bound = 0;
        for (int kb = 0; kb < NKB; kb++) {
            float mp[64], mn[64], sp = NB[i], sn = NB[i];
            for (int sq = 0; sq < 64; sq++) { float x = 0, y = 0; for (int r = kb * 12; r < kb * 12 + 12; r++) { float v = NE[((size_t)r * 64 + sq) * NH + i]; if (v > x) x = v; if (v < y) y = v; } mp[sq] = x; mn[sq] = -y; }
            for (int k = 0; k < 32; k++) {
                int b = k; for (int j = k + 1; j < 64; j++) if (mp[j] > mp[b]) b = j; float t = mp[b]; mp[b] = mp[k]; mp[k] = t; sp += t;
                b = k; for (int j = k + 1; j < 64; j++) if (mn[j] > mn[b]) b = j; t = mn[b]; mn[b] = mn[k]; mn[k] = t; sn -= t;
            }
            if (sp > bound) bound = sp; if (-sn > bound) bound = -sn;
        }
        if (bound > worst) worst = bound;
        int q = 10; while (q > 4 && bound * (float)(1 << q) > 32000.0f) q--;
        QU[i] = (unsigned char)q;
    }
    q_bits = 10; while (q_bits > 4 && worst * (float)(1 << q_bits) > 32000.0f) q_bits--;
    // Default: the single scale set by the worst unit (the standing behaviour; measured level with the finer ones in play).
    // QUNIT=1: every unit its own finest scale. QBITS=n: one scale 2^-n for all (sums wrap; only safe while no unit's final sum leaves 16 bits).
    if (getenv("QBITS") && atoi(getenv("QBITS")) > 0) { q_bits = atoi(getenv("QBITS")); for (int i = 0; i < NH; i++) QU[i] = (unsigned char)q_bits; }
    else if (!getenv("QUNIT")) for (int i = 0; i < NH; i++) QU[i] = (unsigned char)q_bits;
    for (int i = 0; i < HMAX + 16; i++) { int q = i < NH ? QU[i] : q_bits; QU_ONE[i] = (short)(1 << q); QU_MUL[i] = (short)(1 << (15 - q)); }
    for (int i = 0; i < NH; i++) hist_q[QU[i]]++;
    if (getenv("QINFO")) { printf("info string integer scale per unit:"); for (int q = 4; q <= 10; q++) if (hist_q[q]) printf(" %d units with %d steps,", hist_q[q], 1 << q); printf(" worst bound %.1f\n", worst); }
    size_t ne = (size_t)NKB * 768 * NH;
    free(NEq); NEq = (short *)big_alloc(ne * 2);
    for (size_t k = 0; k < ne; k++) NEq[k] = (short)lrintf(NE[k] * (float)(1 << QU[k % NH]));
    for (int i = 0; i < NH; i++) NBq[i] = (short)lrintf(NB[i] * (float)(1 << QU[i]));
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
    if ((hdr[0] == 0x4F555233 || hdr[0] == 0x4F555235 || hdr[0] == 0x4F555236) && (fread(&nk, 4, 1, f) != 1 || nk < 1 || nk > 32)) { fclose(f); return 0; }
    if (hdr[0] == 0x4F555235 && (fread(&nm, 4, 1, f) != 1 || fread(&ng, 4, 1, f) != 1 || nm < 1 || nb % nm || ng < 1 || ng > 64)) { fclose(f); return 0; }
    have_net = 0; NH = hdr[1]; NBK = nb / nm; NKB = nk; NM = nm; NG = ng; KBT = nk == 32 ? KBT32 : nk == 16 ? KBT16 : KBT7;
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
    have_vol = 0; have_pol = 0; have_mlp = 0; NMIR = 0;
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
        else if (tag == 0x3152494D && nk == 1) NMIR = 1;
        else if ((tag == 0x31504C4D || tag == 0x32504C4D) && nm == 1 && NH && NH % 16 == 0) {
            int hh[3], good = fread(hh, 4, 2, f) == 2 && hh[0] >= 8 && hh[0] <= M1MAX && hh[0] % 8 == 0 && hh[1] >= 1 && hh[1] <= M2MAX;
            hh[2] = NH;
            if (good && tag == 0x32504C4D) good = fread(hh + 2, 4, 1, f) == 1 && hh[2] >= 16 && hh[2] <= NH && hh[2] % 16 == 0;
            if (!good) break;
            int h1 = hh[0], h2 = hh[1], nin = 2 * hh[2], nch = nin / 16;
            float *tmp = (float *)malloc((size_t)nb * h1 * nin * 4);
            good &= fread(tmp, 4, (size_t)nb * h1 * nin, f) == (size_t)nb * h1 * nin;
            for (int k = 0; k < nb; k++) good &= fread(MB1[k], 4, h1, f) == (size_t)h1;
            memset(ML2, 0, sizeof ML2); memset(MB2, 0, sizeof MB2); memset(ML3, 0, sizeof ML3);
            for (int k = 0; k < nb; k++) for (int i = 0; i < h2; i++) { float row[M1MAX]; good &= fread(row, 4, h1, f) == (size_t)h1; for (int j = 0; j < h1; j++) ML2[k][j][i] = row[j]; }
            for (int k = 0; k < nb; k++) good &= fread(MB2[k], 4, h2, f) == (size_t)h2;
            for (int k = 0; k < nb; k++) good &= fread(ML3[k], 4, h2, f) == (size_t)h2;
            if (good) {
                free(ML1q); ML1q = (short *)aligned_alloc(64, ((size_t)nb * nch * h1 * 16 * 2 + 63) / 64 * 64);
                for (int k = 0; k < nb; k++) for (int pr = 0; pr < nin / 2; pr++) for (int j = 0; j < h1; j++) for (int l = 0; l < 2; l++) {
                    float v = tmp[((size_t)k * h1 + j) * nin + pr * 2 + l] * 1024.0f; v = v > 32767 ? 32767 : (v < -32767 ? -32767 : v);
                    ML1q[(((size_t)k * (nin / 2) + pr) * h1 + j) * 2 + l] = (short)lrintf(v);
                }
                MH1 = h1; MH2 = h2; MHK = hh[2];
                int h2r = (h2 + 7) & ~7; mlf_stride = (h1 + h1 * h2r + 2 * h2r + 15) & ~15; mlp_dense = nch <= 16;
                free(MLF); MLF = (float *)aligned_alloc(64, (size_t)nb * mlf_stride * 4); memset(MLF, 0, (size_t)nb * mlf_stride * 4);
                for (int k = 0; k < nb; k++) {
                    float *F = MLF + (size_t)k * mlf_stride;
                    for (int j = 0; j < h1; j++) { F[j] = MB1[k][j]; for (int i = 0; i < h2; i++) F[h1 + j * h2r + i] = ML2[k][j][i]; }
                    for (int i = 0; i < h2; i++) { F[h1 + h1 * h2r + i] = MB2[k][i]; F[h1 + h1 * h2r + h2r + i] = ML3[k][i]; }
                }
                if (mlp_dense) {
                    free(MLD); MLD = (short *)aligned_alloc(64, ((size_t)nb * nch * h1 * 16 * 2 + 63) / 64 * 64);
                    for (int k = 0; k < nb; k++) for (int c = 0; c < nch; c++) for (int j = 0; j < h1; j++) for (int l = 0; l < 16; l++) {
                        float v = tmp[((size_t)k * h1 + j) * nin + c * 16 + l] * 1024.0f; v = v > 32767 ? 32767 : (v < -32767 ? -32767 : v);
                        MLD[(((size_t)k * nch + c) * h1 + j) * 16 + l] = (short)lrintf(v);
                    }
                }
            }
            free(tmp); have_mlp = good; if (!good) break;
        }
        else break;
    }
    fclose(f);
    if (!taper) { NC2 = 0; memset(NS2, 0, sizeof NS2); }
    have_net = (int)ok;
    if (have_net && NH) quantize_net();
    kc_reset();
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
static inline long long qdot(const short *a, const short *b, const short *w, int n, int oa, int ob) {      // a[i] belongs to unit oa + i, b[i] to unit ob + i
    long long out = 0; int i = 0;
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    __m256i acc64 = _mm256_setzero_si256();
    while (i + 16 <= n) {
        __m256i acc32 = _mm256_setzero_si256();
        for (int k = 0; k < 4 && i + 16 <= n; k++, i += 16) {          // four steps at most before widening: cannot overflow 32 bits
            __m256i x = _mm256_mullo_epi16(_mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256((const __m256i *)(a + i)), zero), _mm256_loadu_si256((const __m256i *)(QU_ONE + oa + i))), _mm256_loadu_si256((const __m256i *)(QU_MUL + oa + i)));
            __m256i y = _mm256_mullo_epi16(_mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256((const __m256i *)(b + i)), zero), _mm256_loadu_si256((const __m256i *)(QU_ONE + ob + i))), _mm256_loadu_si256((const __m256i *)(QU_MUL + ob + i)));
            acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(_mm256_mulhi_epu16(x, y), _mm256_loadu_si256((const __m256i *)(w + i))));
        }
        acc64 = _mm256_add_epi64(acc64, _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(acc32)), _mm256_cvtepi32_epi64(_mm256_extracti128_si256(acc32, 1))));
    }
    long long t[4]; _mm256_storeu_si256((__m256i *)t, acc64); out = t[0] + t[1] + t[2] + t[3];
#endif
    for (; i < n; i++) {
        int oi = QU_ONE[oa + i], oj = QU_ONE[ob + i];
        unsigned x = (unsigned)(a[i] < 0 ? 0 : (a[i] > oi ? oi : a[i])) * (unsigned)(unsigned short)QU_MUL[oa + i], y = (unsigned)(b[i] < 0 ? 0 : (b[i] > oj ? oj : b[i])) * (unsigned)(unsigned short)QU_MUL[ob + i];
        out += (long long)((x * y) >> 16) * w[i];
    }
    return out;
}
// the layered read-out of bucket bk, in score units (logit of the win probability)
static unsigned short nnz_lut[256][8];      // for every 8-bit mask, the positions of its set bits (the rest: don't care)
static void init_nnz(void) { for (int m = 0; m < 256; m++) { int k = 0; for (int b = 0; b < 8; b++) if (m >> b & 1) nnz_lut[m][k++] = (unsigned short)b; } }
// squared clipped hidden units of both views as 16-bit integers (1.0 = 16384): T[0..n) the mover's, T[n..2n) the opponent's
static inline void act_sq(const short *ao, const short *ap, short *T) {
    int n = NH; int i = 0;
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    for (; i + 16 <= n; i += 16) {
        const __m256i one = _mm256_load_si256((const __m256i *)(QU_ONE + i)), mul = _mm256_load_si256((const __m256i *)(QU_MUL + i));
        __m256i x = _mm256_mullo_epi16(_mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256((const __m256i *)(ao + i)), zero), one), mul);
        __m256i y = _mm256_mullo_epi16(_mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256((const __m256i *)(ap + i)), zero), one), mul);
        _mm256_store_si256((__m256i *)(T + i), _mm256_mulhi_epu16(x, x)); _mm256_store_si256((__m256i *)(T + n + i), _mm256_mulhi_epu16(y, y));
    }
#endif
    for (; i < n; i++) {
        int o = QU_ONE[i];
        unsigned x = (unsigned)(ao[i] < 0 ? 0 : (ao[i] > o ? o : ao[i])) * (unsigned)(unsigned short)QU_MUL[i], y = (unsigned)(ap[i] < 0 ? 0 : (ap[i] > o ? o : ap[i])) * (unsigned)(unsigned short)QU_MUL[i];
        T[i] = (short)((x * x) >> 16); T[n + i] = (short)((y * y) >> 16);
    }
}
// sum of T[i] * w[i]: the direct read-out from the units act_sq prepared (the same integers qdot adds up)
static inline long long tdot(const short *T, const short *w, int n) {
    long long out = 0; int i = 0;
#if defined(__AVX2__)
    __m256i acc64 = _mm256_setzero_si256();
    while (i + 16 <= n) {
        __m256i acc32 = _mm256_setzero_si256();
        for (int k = 0; k < 4 && i + 16 <= n; k++, i += 16) acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(_mm256_load_si256((const __m256i *)(T + i)), _mm256_loadu_si256((const __m256i *)(w + i))));
        acc64 = _mm256_add_epi64(acc64, _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(acc32)), _mm256_cvtepi32_epi64(_mm256_extracti128_si256(acc32, 1))));
    }
    long long t[4]; _mm256_storeu_si256((__m256i *)t, acc64); out = t[0] + t[1] + t[2] + t[3];
#endif
    for (; i < n; i++) out += (long long)T[i] * w[i];
    return out;
}
static float mlp_out(const short *Tall, int bk) {
    int n = MHK, h1 = MH1, h2 = MH2, nch = 2 * n / 16; const short *W = ML1q + (size_t)bk * nch * h1 * 16;
    int acc[M1MAX + 8]; short Tc[2 * HMAX] __attribute__((aligned(32))); const short *T = Tall;
    if (n < NH) { memcpy(Tc, Tall, sizeof(short) * n); memcpy(Tc + n, Tall + NH, sizeof(short) * n); T = Tc; }   // the units the layers read, side by side
#if defined(__AVX2__)
    {
        // first the list of input pairs that are not both zero, without a branch per pair (a branch the processor cannot
        // predict costs more than the arithmetic it would save); then one straight loop over the list
        const __m256i zero = _mm256_setzero_si256(); unsigned short list[HMAX + 16]; int cnt = 0;
        for (int c = 0; c < nch; c++) {
            unsigned m = ~(unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_load_si256((const __m256i *)(T + c * 16)), zero))) & 0xFFu;
            _mm_storeu_si128((__m128i *)(list + cnt), _mm_add_epi16(_mm_loadu_si128((const __m128i *)nnz_lut[m]), _mm_set1_epi16((short)(c * 8))));
            cnt += __builtin_popcount(m);
        }
        __m256i s0 = zero, s1 = zero, s2 = zero, s3 = zero;
        if (h1 == 16) {
            for (int k = 0; k < cnt; k++) {
                int pr = list[k], two; memcpy(&two, T + 2 * pr, 4);        // the pair's two inputs as one 32-bit lane
                const __m256i x = _mm256_set1_epi32(two); const short *w = W + (size_t)pr * 32;
                s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(x, _mm256_loadu_si256((const __m256i *)w)));
                s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(x, _mm256_loadu_si256((const __m256i *)(w + 16))));
            }
        } else if (h1 == 8) {
            for (int k = 0; k < cnt; k++) {
                int pr = list[k], two; memcpy(&two, T + 2 * pr, 4);
                s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_set1_epi32(two), _mm256_loadu_si256((const __m256i *)(W + (size_t)pr * 16))));
            }
        } else {
            const int nv = h1 / 8;
            for (int k = 0; k < cnt; k++) {
                int pr = list[k], two; memcpy(&two, T + 2 * pr, 4);
                const __m256i x = _mm256_set1_epi32(two); const short *w = W + (size_t)pr * h1 * 2;
                s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(x, _mm256_loadu_si256((const __m256i *)w)));
                s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(x, _mm256_loadu_si256((const __m256i *)(w + 16))));
                s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(x, _mm256_loadu_si256((const __m256i *)(w + 32))));
                if (nv > 3) s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(x, _mm256_loadu_si256((const __m256i *)(w + 48))));
            }
        }
        _mm256_storeu_si256((__m256i *)acc, s0); _mm256_storeu_si256((__m256i *)(acc + 8), s1); _mm256_storeu_si256((__m256i *)(acc + 16), s2); _mm256_storeu_si256((__m256i *)(acc + 24), s3);
    }
#else
    for (int j = 0; j < h1; j++) { int a = 0; for (int i = 0; i < 2 * n; i++) a += T[i] * W[((size_t)(i / 2) * h1 + j) * 2 + (i & 1)]; acc[j] = a; }
#endif
    float u[M1MAX], out = 0;
    for (int j = 0; j < h1; j++) { float v = MB1[bk][j] + (float)acc[j] * (1.0f / 16777216.0f); v = v < 0 ? 0 : (v > 1 ? 1 : v); u[j] = v * v; }
#if defined(__AVX2__)
    // second layer: every output unit at once, one input at a time (summing one output after the other is a chain of
    // dependent additions: it cost more than the whole first layer)
    int nv = (h2 + 7) / 8; __m256 v[M2MAX / 8];
    for (int k = 0; k < nv; k++) v[k] = _mm256_load_ps(MB2[bk] + 8 * k);
    for (int j = 0; j < h1; j++) {
        const __m256 uj = _mm256_set1_ps(u[j]); const float *l2 = ML2[bk][j];
        for (int k = 0; k < nv; k++) v[k] = _mm256_add_ps(v[k], _mm256_mul_ps(uj, _mm256_load_ps(l2 + 8 * k)));
    }
    __m256 o = _mm256_setzero_ps(); const __m256 f0 = _mm256_setzero_ps(), f1 = _mm256_set1_ps(1.0f);
    for (int k = 0; k < nv; k++) o = _mm256_add_ps(o, _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(v[k], f0), f1), _mm256_load_ps(ML3[bk] + 8 * k)));
    float t8[8]; _mm256_storeu_ps(t8, o); out = t8[0] + t8[1] + t8[2] + t8[3] + t8[4] + t8[5] + t8[6] + t8[7];
#else
    for (int i = 0; i < h2; i++) {
        float v = MB2[bk][i];
        for (int j = 0; j < h1; j++) v += ML2[bk][j][i] * u[j];
        v = v < 0 ? 0 : (v > 1 ? 1 : v); out += ML3[bk][i] * v;
    }
#endif
    return out;
}
// The layers when they read few units: squared clipped units of just those, first layer over all of them in order (eight
// sums at a time in registers), float layers from one block. The same numbers as mlp_out adds up.
static float mlp_small(const short *ao, const short *ap, int bk) {
    const int n = MHK, h1 = MH1, h2r = (MH2 + 7) & ~7, nch = 2 * n / 16; int acc[M1MAX + 8];
    const short *W = MLD + (size_t)bk * nch * h1 * 16; const float *F = MLF + (size_t)bk * mlf_stride;
#if defined(__AVX2__)
    short T[2 * 128] __attribute__((aligned(32)));
    {
        const __m256i zero = _mm256_setzero_si256();
        for (int i = 0; i < n; i += 16) {
            const __m256i one = _mm256_load_si256((const __m256i *)(QU_ONE + i)), mul = _mm256_load_si256((const __m256i *)(QU_MUL + i));
            __m256i x = _mm256_mullo_epi16(_mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256((const __m256i *)(ao + i)), zero), one), mul);
            __m256i y = _mm256_mullo_epi16(_mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256((const __m256i *)(ap + i)), zero), one), mul);
            _mm256_store_si256((__m256i *)(T + i), _mm256_mulhi_epu16(x, x)); _mm256_store_si256((__m256i *)(T + n + i), _mm256_mulhi_epu16(y, y));
        }
    }
    for (int j0 = 0; j0 < h1; j0 += 8) {
        __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0, s4 = s0, s5 = s0, s6 = s0, s7 = s0;
        const short *w = W + (size_t)j0 * 16;
        for (int c = 0; c < nch; c++, w += (size_t)h1 * 16) {
            const __m256i t = _mm256_load_si256((const __m256i *)(T + c * 16));
            s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(t, _mm256_load_si256((const __m256i *)(w))));
            s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(t, _mm256_load_si256((const __m256i *)(w + 16))));
            s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(t, _mm256_load_si256((const __m256i *)(w + 32))));
            s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(t, _mm256_load_si256((const __m256i *)(w + 48))));
            s4 = _mm256_add_epi32(s4, _mm256_madd_epi16(t, _mm256_load_si256((const __m256i *)(w + 64))));
            s5 = _mm256_add_epi32(s5, _mm256_madd_epi16(t, _mm256_load_si256((const __m256i *)(w + 80))));
            s6 = _mm256_add_epi32(s6, _mm256_madd_epi16(t, _mm256_load_si256((const __m256i *)(w + 96))));
            s7 = _mm256_add_epi32(s7, _mm256_madd_epi16(t, _mm256_load_si256((const __m256i *)(w + 112))));
        }
        __m256i a01 = _mm256_hadd_epi32(s0, s1), a23 = _mm256_hadd_epi32(s2, s3), a45 = _mm256_hadd_epi32(s4, s5), a67 = _mm256_hadd_epi32(s6, s7);
        __m256i b03 = _mm256_hadd_epi32(a01, a23), b47 = _mm256_hadd_epi32(a45, a67);
        _mm_storeu_si128((__m128i *)(acc + j0), _mm_add_epi32(_mm256_castsi256_si128(b03), _mm256_extracti128_si256(b03, 1)));
        _mm_storeu_si128((__m128i *)(acc + j0 + 4), _mm_add_epi32(_mm256_castsi256_si128(b47), _mm256_extracti128_si256(b47, 1)));
    }
    float u[M1MAX];
    for (int j = 0; j < h1; j++) { float v = F[j] + (float)acc[j] * (1.0f / 16777216.0f); v = v < 0 ? 0 : (v > 1 ? 1 : v); u[j] = v * v; }
    const float *l2 = F + h1, *b2 = F + h1 + h1 * h2r, *l3 = b2 + h2r; const __m256 f0 = _mm256_setzero_ps(), f1 = _mm256_set1_ps(1.0f); __m256 o = _mm256_setzero_ps();
    if (h2r == 32) {                                                     // the usual size: the four sums stay in registers
        __m256 v0 = _mm256_load_ps(b2), v1 = _mm256_load_ps(b2 + 8), v2 = _mm256_load_ps(b2 + 16), v3 = _mm256_load_ps(b2 + 24);
        for (int j = 0; j < h1; j++, l2 += 32) {
            const __m256 uj = _mm256_set1_ps(u[j]);
            v0 = _mm256_add_ps(v0, _mm256_mul_ps(uj, _mm256_load_ps(l2))); v1 = _mm256_add_ps(v1, _mm256_mul_ps(uj, _mm256_load_ps(l2 + 8)));
            v2 = _mm256_add_ps(v2, _mm256_mul_ps(uj, _mm256_load_ps(l2 + 16))); v3 = _mm256_add_ps(v3, _mm256_mul_ps(uj, _mm256_load_ps(l2 + 24)));
        }
        o = _mm256_add_ps(o, _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(v0, f0), f1), _mm256_load_ps(l3)));
        o = _mm256_add_ps(o, _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(v1, f0), f1), _mm256_load_ps(l3 + 8)));
        o = _mm256_add_ps(o, _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(v2, f0), f1), _mm256_load_ps(l3 + 16)));
        o = _mm256_add_ps(o, _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(v3, f0), f1), _mm256_load_ps(l3 + 24)));
    } else {
        int nv = h2r / 8; __m256 v[M2MAX / 8];
        for (int k = 0; k < nv; k++) v[k] = _mm256_load_ps(b2 + 8 * k);
        for (int j = 0; j < h1; j++, l2 += h2r) { const __m256 uj = _mm256_set1_ps(u[j]); for (int k = 0; k < nv; k++) v[k] = _mm256_add_ps(v[k], _mm256_mul_ps(uj, _mm256_load_ps(l2 + 8 * k))); }
        for (int k = 0; k < nv; k++) o = _mm256_add_ps(o, _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(v[k], f0), f1), _mm256_load_ps(l3 + 8 * k)));
    }
    float t8[8]; _mm256_storeu_ps(t8, o);
    return t8[0] + t8[1] + t8[2] + t8[3] + t8[4] + t8[5] + t8[6] + t8[7];
#else
    (void)W; (void)F; (void)acc; (void)h2r; (void)nch; (void)ao; (void)ap; return 0;
#endif
}
#ifdef SFNET
// DIAGNOSTIC BUILD ONLY (-DSFNET, never shipped): the installed Stockfish 14.1's own network as this engine's evaluator, to
// measure the two searches with the evaluator held equal. The weights are read from the file named in SFNET_FILE (the
// network embedded in the installed binary, extracted by ext/sfnet_extract.py); this is our own re-implementation of the
// arithmetic, checked against the engine's `eval` output on 2,000 positions (ext/sfnet_ref.py). Recomputed from scratch at
// every call: slow, so it is only used at equal node budgets.
static short *sf_W, sf_bias[1024]; static int *sf_PS; static int sf_b0[8][8], sf_b1[8][32], sf_b2[8]; static signed char sf_w0[8][8][2048], sf_w1[8][32][32], sf_w2[8][32];
static int sf_ok = 0;
static int sf_load(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    unsigned hdr[3]; if (fread(hdr, 4, 3, f) != 3 || hdr[0] != 0x7AF32F20u) { fclose(f); return 0; }
    fseek(f, hdr[2] + 4, SEEK_CUR);
    sf_W = (short *)big_alloc((size_t)22528 * 1024 * 2); sf_PS = (int *)big_alloc((size_t)22528 * 8 * 4);
    int ok = fread(sf_bias, 2, 1024, f) == 1024 && fread(sf_W, 2, (size_t)22528 * 1024, f) == (size_t)22528 * 1024 && fread(sf_PS, 4, 22528 * 8, f) == 22528 * 8;
    for (int k = 0; k < 8 && ok; k++) {
        unsigned h; ok = fread(&h, 4, 1, f) == 1 && fread(sf_b0[k], 4, 8, f) == 8 && fread(sf_w0[k], 1, 8 * 2048, f) == 8 * 2048 &&
             fread(sf_b1[k], 4, 32, f) == 32 && fread(sf_w1[k], 1, 32 * 32, f) == 32 * 32 && fread(&sf_b2[k], 4, 1, f) == 1 && fread(sf_w2[k], 1, 32, f) == 32;
    }
    fclose(f); return sf_ok = ok;
}
static int sf_raw(int adjusted) {                 // Stockfish's internal units, from the mover's point of view
    static const int plane[6] = {0, 2, 4, 6, 8, 10};
    int acc[2][1024] __attribute__((aligned(64))); int psq[2][8], np = 0;
    for (int v = 0; v < 2; v++) {
        int ksq = lsb(P.bb[v][KING]), flip = v == WHITE ? 0 : 56, mir = (((ksq ^ flip) & 7) < 4) ? 7 : 0, ko = ksq ^ flip ^ mir;
        int kb = (7 - (ko >> 3)) * 4 + (7 - (ko & 7));
        for (int j = 0; j < 1024; j++) acc[v][j] = sf_bias[j];
        for (int k = 0; k < 8; k++) psq[v][k] = 0;
        for (int c = 0; c < 2; c++) for (int pt = 0; pt < 6; pt++) {
            U64 b = P.bb[c][pt];
            while (b) {
                int sq = lsb(b); b &= b - 1; if (v == 0) np++;
                int idx = (sq ^ flip ^ mir) + (plane[pt] + ((c == v || pt == KING) ? 0 : 1)) * 64 + 704 * kb;
                const short *w = sf_W + (size_t)idx * 1024; for (int j = 0; j < 1024; j++) acc[v][j] += w[j];
                const int *q = sf_PS + (size_t)idx * 8; for (int k = 0; k < 8; k++) psq[v][k] += q[k];
            }
        }
    }
    int us = P.side, bk = (np - 1) / 4, x[2048];
    for (int j = 0; j < 1024; j++) { int a = acc[us][j], b2 = acc[us ^ 1][j]; x[j] = a < 0 ? 0 : (a > 127 ? 127 : a); x[1024 + j] = b2 < 0 ? 0 : (b2 > 127 ? 127 : b2); }
    int h0[8], h1[32];
    for (int i = 0; i < 8; i++) { int t = sf_b0[bk][i]; const signed char *w = sf_w0[bk][i]; for (int j = 0; j < 2048; j++) t += w[j] * x[j]; t >>= 6; h0[i] = t < 0 ? 0 : (t > 127 ? 127 : t); }
    for (int i = 0; i < 32; i++) { int t = sf_b1[bk][i]; for (int j = 0; j < 8; j++) t += sf_w1[bk][i][j] * h0[j]; t >>= 6; h1[i] = t < 0 ? 0 : (t > 127 ? 127 : t); }
    int out = sf_b2[bk]; for (int j = 0; j < 32; j++) out += sf_w2[bk][j] * h1[j];
    int ps = (psq[us][bk] - psq[us ^ 1][bk]) / 2;
    if (adjusted) {
        static const int mg[5] = {0, 781, 825, 1276, 2538}; int npm[2] = {0, 0};
        for (int c = 0; c < 2; c++) for (int pt = 1; pt < 5; pt++) npm[c] += mg[pt] * popc(P.bb[c][pt]);
        if (abs(npm[0] - npm[1]) <= 1276 - 825) return ((128 - 7) * ps + (128 + 7) * out) / 128 / 16;
    }
    return (ps + out) / 16;
}
static int sf_eval(void) {                        // as Stockfish's search sees it (material scaling, shuffling damp), in this engine's units
    static const int mg[5] = {0, 781, 825, 1276, 2538}; int npm = 0;
    for (int c = 0; c < 2; c++) for (int pt = 1; pt < 5; pt++) npm += mg[pt] * popc(P.bb[c][pt]);
    int scale = 883 + 32 * popc(P.bb[0][PAWN] | P.bb[1][PAWN]) + 32 * npm / 1024;
    int v = sf_raw(1) * scale / 1024; v = v * (100 - (P.half > 100 ? 100 : P.half)) / 100;
    return v * 277 / 1000;                                         // 361 internal units = 100 of ours (the scale of the Lichess labels)
}
#endif
#ifdef TSTAT                              // where the evaluator's time goes (printed at quit): -DTSTAT, test builds only
static unsigned long long ts_acc, ts_out, ts_n, ts_start, ts_nodes, ts_go;
#endif
static int evaluate(void) {
    int us = P.side;
#ifdef SFNET
    if (sf_ok) return sf_eval();
#endif
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
#ifdef TSTAT
    unsigned long long ts0 = __rdtsc();
#endif
    acc_ensure();
#ifdef TSTAT
    unsigned long long ts1 = __rdtsc(); ts_acc += ts1 - ts0; ts_n++;
#endif
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
    if (NBK > 1) {                       // read-out bucket: piece count; with 16 buckets also whether a queen is on the board, with 32 whose
        bk = (popc(P.occ[0] | P.occ[1]) - 2) / 4; if (bk > 7) bk = 7;
        if (NBK == 16) bk = bk * 2 + ((P.bb[0][QUEEN] | P.bb[1][QUEEN]) != 0);
        else if (NBK == 32) bk = bk * 4 + 2 * (P.bb[P.side][QUEEN] != 0) + (P.bb[P.side ^ 1][QUEEN] != 0);
        else if (bk > NBK - 1) bk = NBK - 1;
    }
    if (NM > 1) {                        // learned router: the piece-count group has NM specialists; run the one the router scores highest
        float x[128]; int best = 0; float bg = -1e30f;
        for (int i = 0; i < NG; i++) { const float qi = 1.0f / (float)QU_ONE[i]; float a = ao[i] * qi, b = ap[i] * qi; a = a < 0 ? 0 : (a > 1 ? 1 : a); b = b < 0 ? 0 : (b > 1 ? 1 : b); x[i] = a * a; x[NG + i] = b * b; }
        for (int m = 0; m < NM; m++) {
            const float *gw = NGW[bk * NM + m]; float g = NGB[bk * NM + m];
            for (int i = 0; i < 2 * NG; i++) g += gw[i] * x[i];
            if (g > bg) { bg = g; best = m; }
        }
        bk = bk * NM + best;
    }
    short T[2 * HMAX] __attribute__((aligned(32))); long long q;
    int small = 0;
#if defined(__AVX2__)
    small = have_mlp && mlp_dense;
#endif
    if (small) {                         // fetch the layers' two blocks now: they are in the nearest cache by the time the direct read-out is done
        const char *pf = (const char *)(MLF + (size_t)bk * mlf_stride), *pw = (const char *)(MLD + (size_t)bk * (2 * MHK / 16) * MH1 * 16);
        for (int o = 0; o < mlf_stride * 4; o += 64) __builtin_prefetch(pf + o);
        for (int o = 0; o < (2 * MHK / 16) * MH1 * 32; o += 64) __builtin_prefetch(pw + o);
        q = qdot(ao, ao, NWOq[bk], n, 0, 0) + qdot(ap, ap, NWPq[bk], n, 0, 0);
    }
    else if (have_mlp) { act_sq(ao, ap, T); q = tdot(T, NWOq[bk], n) + tdot(T + n, NWPq[bk], n); }
    else q = qdot(ao, ao, NWOq[bk], n, 0, 0) + qdot(ap, ap, NWPq[bk], n, 0, 0);
    if (NPROD) q += qdot(ao, ao + n / 2, NQOq[bk], n / 2, 0, n / 2) + qdot(ap, ap + n / 2, NQPq[bk], n / 2, 0, n / 2) + qdot(ao, ap, NQXq[bk], n, 0, 0);
    float out = lin + NCB[bk] + (float)q * (1.0f / 33554432.0f);
    if (have_mlp) out += small ? mlp_small(ao, ap, bk) : mlp_out(T, bk);
    int v = (int)(out * 271.58f);        // logit of win probability -> centipawns
#ifdef TSTAT
    ts_out += __rdtsc() - ts1;
#endif
    return v > 20000 ? 20000 : (v < -20000 ? -20000 : v);
}

// How far is a search likely to move the static score of this position? In centipawns, read off the same hidden layer
// (call after evaluate(): the sums are in place). Lets the search prune with narrow margins where the evaluator is
// reliable and wide ones where it is not.
static int eval_vol(void) {
    const Acc *A = &accs[hply]; int us = P.side, bk = 0;
    if (NBK > 1) {                       // read-out bucket: piece count; with 16 buckets also whether a queen is on the board, with 32 whose
        bk = (popc(P.occ[0] | P.occ[1]) - 2) / 4; if (bk > 7) bk = 7;
        if (NBK == 16) bk = bk * 2 + ((P.bb[0][QUEEN] | P.bb[1][QUEEN]) != 0);
        else if (NBK == 32) bk = bk * 4 + 2 * (P.bb[P.side][QUEEN] != 0) + (P.bb[P.side ^ 1][QUEEN] != 0);
        else if (bk > NBK - 1) bk = NBK - 1;
    }
    float u = NV[bk][2 * NH] + (float)(qdot(A->a[us], A->a[us], NVOq[bk], NH, 0, 0) + qdot(A->a[us ^ 1], A->a[us ^ 1], NVPq[bk], NH, 0, 0)) * (1.0f / 33554432.0f);
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
#define NPAR 62
enum { RAZOR, RFP, NMP_BASE, NMP_DIV, NMP_EV, LMP_BASE, FUT_BASE, FUT_MUL, HIST_PRUNE, SEE_Q, SEE_C, SE_DEPTH, SE_MUL, ASP, DELTA_Q, LMR_BASE, LMR_DIV, HB_QUAD, HB_LIN, HB_MAX, PROBCUT, SE_DOUBLE, CUT_LMR, CUT_IIR, NEG_EXT, TTCAP_LMR, LMR_HIST, CAP_LMR, DEEPER, NP_CORR, CHK_EXT, VOL_MIX, VOL_REF, VOL_LMR, VOL_NMP, CONT_CORR, NMP_TT, PV_PRUNE, THR_HIST, VOL_TM, ROOT_LMR, ROOT_ORD, POL_W, POL_DEPTH, POL_LMR, TT_BUCKET, MM_CORR, THR_RFP, THR_LMR, LMR_PRUNE, LMP_DEPTH, NMP_MARGIN, EVAL_SCALE, HIST_V2, PRUNE_V2, SEARCH_V3, SEL, SEL_A, SEL_B, OFF, LM_DEPTH, LM_WAIT };
static const char *par_name[NPAR] = {"Razor", "Rfp", "NmpBase", "NmpDiv", "NmpEv", "LmpBase", "FutBase", "FutMul", "HistPrune", "SeeQ", "SeeC", "SeDepth", "SeMul", "Asp", "DeltaQ", "LmrBase", "LmrDiv", "HbQuad", "HbLin", "HbMax", "ProbCut", "SeDouble", "CutLmr", "CutIir", "NegExt", "TtCapLmr", "LmrHist", "CapLmr", "Deeper", "NpCorr", "ChkExt", "VolMix", "VolRef", "VolLmr", "VolNmp", "ContCorr", "NmpTt", "PvPrune", "ThrHist", "VolTm", "RootLmr", "RootOrd", "PolW", "PolDepth", "PolLmr", "TtBucket", "MmCorr", "ThrRfp", "ThrLmr", "LmrPrune", "LmpDepth", "NmpMargin", "EvalScale", "HistV2", "PruneV2", "SearchV3", "Sel", "SelA", "SelB", "Off", "LmDepth", "LmWait"};
// search parameters the engine tuned for itself: 20,000 self-play iterations (zero/spsa.py). Against the hand-set values
// (250, 70, 3, 4, 200, 3, 90, 85, 4000, 50, 90, 7, 2, 25, 200, 75, 225, 4, 20, 1200) they scored +50 Elo at 2,500 nodes
// and +37 at 10,000 nodes over 2,000 games each.
static int par[NPAR] = {293, 53, 3, 4, 191, 6, 89, 94, 4245, 48, 92, 7, 2, 33, 190, 68, 240, 5, 33, 1322, 0, 0, 0, 0, 0, 0, 8192, 1, 1, 1, 1, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 5, 0, 100, 0, 0, 0, 0, 250, 100, 0, 0, 1};
// PruneV2 (bit flags, 0 = off): 1 = a late quiet move whose two continuation histories are both negative is skipped when
// its reduced depth is under 5; 2 = quiet futility counts the reduced depth and the move's history; 4 = a quiet table move
// that cuts off straight from the table is rewarded in the histories; 8 = one ply less reduction when the parent node is
// already past its 13th move (it is failing low, this node is likely to cut); 16 = a capture with a negative capture
// history is skipped at reduced depth under 6 outside PV nodes.
// Off (bit flags, for diagnosis only): 1 = no null move, 2 = no reverse futility, 4 = no razoring, 8 = no late-move reductions,
// 16 = no late-move pruning, 32 = no futility pruning of quiet moves, 64 = no losing-exchange pruning, 128 = no bad-history
// pruning, 256 = no singular tests, 512 = no depth cut without a table move, 1024 = no delta pruning in quiescence.
// SearchV3 (bit flags, 0 = off): 1 = a quiet move that gives check is never dropped by the futility or the bad-history rule
// (only by the losing-exchange rule, with the margin captures get); at a futile node the checking quiet moves are still tried.
// 2 = a position that is or has been on a main line (remembered in the hash table) has its late moves reduced one ply less;
// 4 = a late capture is skipped outside PV nodes when even winning the piece leaves the static evaluation well under alpha
// at its reduced depth; 8 = quiescence: standing well under alpha, a capture that wins nothing in the exchange is skipped;
// 16 = a node where nothing raised alpha keeps the hash table's move for this position if there is one (32: and stores no
// move otherwise); 64 = a result much deeper than the new one (4 plies or more) for the same position is kept unless the
// new one is exact; a new cutoff move is still written.
// HistV2 (bit flags, 0 = off): 1 = continuation histories four and six plies back as well, the one-ply one counted twice
// in move ordering; 2 = a counter-move slot after the killers; 4 = the best move of a node that raised alpha without a
// cutoff is rewarded like a cutoff move; 8 = when every move of a node fails low, the opponent's quiet move that led here
// is rewarded; 16 = when a node cuts off, the opponent's quiet move that led here is penalised if it was its node's first.
// EvalScale: the network's value in percent as the search sees it (a network trained on labels of another scale can be
// brought to the scale the search margins were tuned for without retraining).
// LmrPrune 1: the low-depth pruning rules for quiet moves (futility, losing exchange, bad history) count the depth the move
// would be searched to after its late-move reduction, not the node's depth, so they also act on late moves high in the tree.
// LmpDepth: late-move pruning up to this depth. NmpMargin m: a null move is tried only when the static evaluation is at
// least beta + m - m * depth / 10 (at low depth a null move that starts barely above beta mostly fails).
static U64 nodes, node_limit, check_mask = 2047;
#ifdef SSTAT                               // where the search spends its nodes (printed at quit); no effect on the search
static U64 st_[40];
static U64 st_e[2][6];        // evaluations by how far outside the window they fall (0 inside, then by 50, 100, 150, 200, 300+), quiescence and main
static U64 st_d[24], st_dm[24], st_dq[24], st_p[8]; static int st_mode;   // st_p: nodes by what the search was doing (0 first look, 1 inside a null-move try, 2 a singular test, 3 a re-search after a reduced look liked the move, 4 a full-window re-search, 5 a reduced look)
#define STM(m, code) { int o_ = st_mode; st_mode = (m); code; st_mode = o_; }
#else
#define STM(m, code) { code; }
#endif
#ifdef SSTAT        // main-search nodes by remaining depth; moves searched there; quiet moves among them
#define ST(i) (st_[i]++)
#define STA(i, v) (st_[i] += (U64)(v))
#else
#define ST(i) ((void)0)
#define STA(i, v) ((void)0)
#endif
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
    int v0 = b[0].depth - 8 * ((tt_age - (b[0].flag >> 3)) & 31), v1 = b[1].depth - 8 * ((tt_age - (b[1].flag >> 3)) & 31);
    return v1 < v0 ? b + 1 : b;
}
#define TTF(fl) ((unsigned char)((fl) | (tt_age << 3)))     // bits 0-1: the bound; bit 2: the position has been on a main line; then the age
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
static short ch4[12][64][12][64], ch6[12][64][12][64];   // the same four and six plies back (option HistV2)
static Move cmove[12][64];                 // the quiet move that last refuted [piece][target] (option HistV2)
static int mcount[MAXPLY + 4];             // which move of its node (1 = first) is being searched at each ply
static short caph[12][64][6];              // capture history: [piece][target][captured type]
static int corrhist[2][16384];             // correction history: how wrong the static evaluation was for this pawn structure (x256)
static int corr_np[2][2][16384];           // the same, keyed by where one side's pieces (not pawns) stand: [side to move][whose pieces]
static int corr_cont[2][12][64];           // the same, keyed by the move that led to this position: [side to move][piece][target square]
static int corr_mm[2][2][16384];           // the same, keyed by the minor pieces and kings / the major pieces and kings (option MmCorr, weight in 1/256)
static int corr_ply;                       // the ply of the position being evaluated (set by the callers of eval_corr)
static inline int eval_corr(void) {
    int c = corrhist[P.side][P.pkey & 16383];
    if (par[NP_CORR]) c = (2 * c + corr_np[P.side][0][P.npkey[0] & 16383] + corr_np[P.side][1][P.npkey[1] & 16383]) / 4;
    if (par[CONT_CORR] && corr_ply > 0 && move_stack[corr_ply - 1]) c = (3 * c + corr_cont[P.side][piece_stack[corr_ply - 1]][TO(move_stack[corr_ply - 1])]) / 4;
    if (par[MM_CORR]) c += (corr_mm[P.side][0][P.mmkey[0] & 16383] + corr_mm[P.side][1][P.mmkey[1] & 16383]) * par[MM_CORR] / 256;
    int v = evaluate() * par[EVAL_SCALE] / 100 + c / 256;
    return v > 20000 ? 20000 : (v < -20000 ? -20000 : v);
}
static inline int hist_score(Move m, int ply) {
    int h = HIST(m, ply), pc = (int)P.sq[FROM(m)], to = TO(m), v2 = par[HIST_V2] & 1;
    if (ply > 0 && move_stack[ply - 1]) h += cmh[piece_stack[ply - 1]][TO(move_stack[ply - 1])][pc][to] * (1 + v2);
    if (ply > 1 && move_stack[ply - 2]) h += fuh[piece_stack[ply - 2]][TO(move_stack[ply - 2])][pc][to];
    if (v2) {
        if (ply > 3 && move_stack[ply - 4]) h += ch4[piece_stack[ply - 4]][TO(move_stack[ply - 4])][pc][to];
        if (ply > 5 && move_stack[ply - 6]) h += ch6[piece_stack[ply - 6]][TO(move_stack[ply - 6])][pc][to];
    }
    return h;
}
// the continuation histories of a move (piece pc to square to) made at stack index p
static inline void cont_update(int p, int pc, int to, int bonus) {
    if (p > 0 && move_stack[p - 1]) { short *c = &cmh[piece_stack[p - 1]][TO(move_stack[p - 1])][pc][to]; *c += bonus - *c * abs(bonus) / 16384; }
    if (p > 1 && move_stack[p - 2]) { short *c = &fuh[piece_stack[p - 2]][TO(move_stack[p - 2])][pc][to]; *c += bonus - *c * abs(bonus) / 16384; }
    if (par[HIST_V2] & 1) {
        if (p > 3 && move_stack[p - 4]) { short *c = &ch4[piece_stack[p - 4]][TO(move_stack[p - 4])][pc][to]; *c += bonus - *c * abs(bonus) / 16384; }
        if (p > 5 && move_stack[p - 6]) { short *c = &ch6[piece_stack[p - 6]][TO(move_stack[p - 6])][pc][to]; *c += bonus - *c * abs(bonus) / 16384; }
    }
}
static inline void hist_update(Move m, int ply, int bonus) {
    int *h = &HIST(m, ply);
    *h += bonus - *h * abs(bonus) / 16384;
    cont_update(ply, (int)P.sq[FROM(m)], TO(m), bonus);
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
// ------------------------------------------------------------------ the transformer at the top of the tree
// LmDepth=d (and the environment variable OURO_SOCK = the socket of lm/serve.py): at nodes with d or more plies left, the
// quiet moves are scored by the transformer's policy (logits x256, only differences matter; the server sends them x100) in place of the read-out's
// (the same use as the policy read-out: PolW orders them, PolLmr shifts their reductions). The request carries the board
// already encoded in the transformer's frame and the quiet moves' indices, so the server does no chess work:
//   "Q <key hex> <64 piece codes 0-c, mover's frame> <castling bits, mover's frame> <ep square or 64> <n> <move index>..."
// and the reply is "<key hex> <logit x100, integer>..." in the same order. LmWait=1: wait for the reply. LmWait=0: send it and search on;
// the answer is taken from the table when the node is visited again (the next iteration of the deepening loop).
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
static int lm_fd = -2; static unsigned long long lm_calls = 0, lm_hits = 0, lm_late = 0;
#define LMC_BITS 15
#define LMN 96
typedef struct { U64 key; unsigned char state; unsigned short n, mv[LMN]; short lp[LMN]; } LmEnt;   // state: 0 empty, 1 asked, 2 answered
static LmEnt *lm_cache = 0; static char lm_buf[1 << 16]; static int lm_len = 0;
static int lm_ready(void) {
    if (lm_fd == -2) {
        lm_fd = -1; const char *path = getenv("OURO_SOCK");
        if (path && *path) {
            int fd = socket(AF_UNIX, SOCK_STREAM, 0); struct sockaddr_un a; memset(&a, 0, sizeof a); a.sun_family = AF_UNIX; strncpy(a.sun_path, path, sizeof a.sun_path - 1);
            if (fd >= 0 && connect(fd, (struct sockaddr *)&a, sizeof a) == 0) { lm_fd = fd; lm_cache = calloc((size_t)1 << LMC_BITS, sizeof(LmEnt)); }
            else { if (fd >= 0) close(fd); fprintf(stderr, "cannot reach the transformer at %s\n", path); }
        }
    }
    return lm_fd >= 0;
}
static LmEnt *lm_slot(U64 key) { return &lm_cache[key & (((U64)1 << LMC_BITS) - 1)]; }
static void lm_line(char *l) {                  // one reply: fill its table entry if it is still waiting for it
    char *q; U64 key = strtoull(l, &q, 16); LmEnt *e = lm_slot(key);
    if (e->key != key || e->state != 1) { lm_late++; return; }
    for (int j = 0; j < e->n; j++) { float x = strtof(q, &q) * 2.56f; e->lp[j] = (short)(x < -32000.0f ? -32000.0f : (x > 32000.0f ? 32000.0f : x)); }
    e->state = 2;
}
static void lm_read(int block) {                // take in whatever replies have arrived (block: wait for at least some bytes)
    for (;;) {
        ssize_t r = recv(lm_fd, lm_buf + lm_len, sizeof lm_buf - 1 - lm_len, block ? 0 : MSG_DONTWAIT);
        if (r <= 0) { if (r == 0) lm_fd = -1; return; }
        lm_len += (int)r; lm_buf[lm_len] = 0; char *st = lm_buf, *nl;
        while ((nl = strchr(st, '\n'))) { *nl = 0; lm_line(st); st = nl + 1; }
        lm_len -= (int)(st - lm_buf); memmove(lm_buf, st, lm_len);
        if (block) return;
    }
}
static void lm_ask(LmEnt *e, const Move *list, int n) {
    int us = P.side; char req[64 * 2 + 16 * LMN + 128], *o = req;
    o += sprintf(o, "Q %llx ", (unsigned long long)P.key);
    for (int sq = 0; sq < 64; sq++) {
        int pc = P.sq[us == WHITE ? sq : sq ^ 56];
        int c = pc == EMPTY ? 0 : (pc / 6 == us ? 1 + pc % 6 : 7 + pc % 6); *o++ = "0123456789abc"[c];
    }
    int ca = us == WHITE ? P.castle : (((P.castle >> 2) & 3) | ((P.castle & 3) << 2));
    int ep = P.ep == NOSQ ? 64 : (us == WHITE ? P.ep : P.ep ^ 56);
    if (n > LMN) n = LMN;
    o += sprintf(o, " %d %d %d", ca, ep, n);
    e->key = P.key; e->state = 1; e->n = (unsigned short)n;
    for (int k = 0; k < n; k++) {
        int f = FROM(list[k]), t = TO(list[k]); e->mv[k] = (unsigned short)(f * 64 + t);
        if (us == BLACK) { f ^= 56; t ^= 56; }
        o += sprintf(o, " %d", f * 64 + t);
    }
    *o++ = '\n';
    if (send(lm_fd, req, (size_t)(o - req), MSG_NOSIGNAL) < 0) { lm_fd = -1; e->state = 0; return; }
    lm_calls++;
}
static unsigned long long lm_waits = 0, lm_wait_us = 0;
static void lm_early(void) {                                // LmWait=2: the request goes out as the node is entered; the answer is needed only
    LmEnt *e = lm_slot(P.key);                              // when its quiet moves come up, after the table move, the captures, the null move...
    if (e->key == P.key && e->state) return;
    lm_read(0); if (lm_fd < 0) return;
    Move l[256]; int n = gen(l, 2); if (n) lm_ask(e, l, n);
}
static const LmEnt *lm_get(const Move *list, int n) {      // the transformer's scores for this position's quiet moves, if available
    lm_read(0); if (lm_fd < 0) return 0;
    LmEnt *e = lm_slot(P.key);
    if (e->key == P.key && e->state == 2) { lm_hits++; return e; }
    if (!(e->key == P.key && e->state == 1)) lm_ask(e, list, n);
    if (par[LM_WAIT] && lm_fd >= 0 && e->key == P.key && e->state == 1) {
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0); lm_waits++;
        while (lm_fd >= 0 && e->key == P.key && e->state == 1) lm_read(1);
        clock_gettime(CLOCK_MONOTONIC, &t1); lm_wait_us += (unsigned long long)((t1.tv_sec - t0.tv_sec) * 1000000LL + (t1.tv_nsec - t0.tv_nsec) / 1000);
    }
    return (lm_fd >= 0 && e->key == P.key && e->state == 2) ? e : 0;
}
static void pos_fen(char *o) {
    const char *pcs = "PNBRQKpnbrqk";
    for (int r = 7; r >= 0; r--) {
        int emp = 0;
        for (int f = 0; f < 8; f++) { int pc = P.sq[r * 8 + f]; if (pc == EMPTY) emp++; else { if (emp) *o++ = (char)('0' + emp); emp = 0; *o++ = pcs[pc]; } }
        if (emp) *o++ = (char)('0' + emp);
        if (r) *o++ = '/';
    }
    *o++ = ' '; *o++ = P.side == WHITE ? 'w' : 'b'; *o++ = ' ';
    if (!P.castle) *o++ = '-';
    else { if (P.castle & 1) *o++ = 'K'; if (P.castle & 2) *o++ = 'Q'; if (P.castle & 4) *o++ = 'k'; if (P.castle & 8) *o++ = 'q'; }
    *o++ = ' '; if (P.ep == NOSQ) *o++ = '-'; else { *o++ = (char)('a' + (P.ep & 7)); *o++ = (char)('1' + (P.ep >> 3)); }
    sprintf(o, " %d 1", P.half);
}
static void pol_hidden(float *hid) {
    const Acc *A = &accs[hply]; int us = P.side; float h[2 * HMAX];
    for (int v = 0; v < 2; v++) {
        const short *a = A->a[v ? us ^ 1 : us]; float *d = h + v * NH;
        for (int i = 0; i < NH; i++) { float x = a[i] / (float)QU_ONE[i]; x = x < 0 ? 0 : (x > 1 ? 1 : x); d[i] = x * x; }
    }
    for (int k = 0; k < NPH; k++) { float t = PB1[k] + fdot(PW1 + (size_t)k * 2 * NH, h, 2 * NH); hid[k] = t > 0 ? t : 0; }
}
// does the quiet move m (no capture, no promotion) give check? Exact, a little slow; castling is not looked at.
static int gives_check_quiet(Move m) {
    if (FLAG(m) == F_CASTLE) return 0;
    int us = P.side, f = FROM(m), t = TO(m), pt = P.sq[f] % 6, ks = lsb(P.bb[us ^ 1][KING]);
    U64 occ = ((P.occ[0] | P.occ[1]) ^ BIT(f)) | BIT(t), kb = BIT(ks);
    if (pt == PAWN ? (pawn_att[us][t] & kb) != 0 : pt == KNIGHT ? (knight_att[t] & kb) != 0 : pt == BISHOP ? (bishop_att(t, occ) & kb) != 0 :
        pt == ROOK ? (rook_att(t, occ) & kb) != 0 : pt == QUEEN ? ((bishop_att(t, occ) | rook_att(t, occ)) & kb) != 0 : 0) return 1;
    if (bishop_att(ks, occ) & (P.bb[us][BISHOP] | P.bb[us][QUEEN]) & ~BIT(f)) return 1;      // a line opened by leaving f
    if (rook_att(ks, occ) & (P.bb[us][ROOK] | P.bb[us][QUEEN]) & ~BIT(f)) return 1;
    return 0;
}
// once per node: the squares from which each piece type of the side to move checks, and its pieces standing between one of
// its sliders and the enemy king (a few too many: every blocker on a line kind that has such a slider)
static void check_info(U64 *cs, U64 *dc) {
    int us = P.side, ks = lsb(P.bb[us ^ 1][KING]); U64 occ = P.occ[0] | P.occ[1];
    U64 b = bishop_att(ks, occ), r = rook_att(ks, occ), bl = b & P.occ[us], rl = r & P.occ[us], d = 0;
    cs[PAWN] = pawn_att[us ^ 1][ks]; cs[KNIGHT] = knight_att[ks]; cs[BISHOP] = b; cs[ROOK] = r; cs[QUEEN] = b | r; cs[KING] = 0;
    if (bl && (bishop_att(ks, occ ^ bl) & ~b & (P.bb[us][BISHOP] | P.bb[us][QUEEN]))) d |= bl;
    if (rl && (rook_att(ks, occ ^ rl) & ~r & (P.bb[us][ROOK] | P.bb[us][QUEEN]))) d |= rl;
    *dc = d;
}
enum { ST_TT, ST_GENCAP, ST_CAP, ST_GENQUIET, ST_QUIET, ST_BAD, ST_DONE };
typedef struct {
    Move list[256], bad[96], ttm; int sc[256], bsc[96], bsee[96];
    int n, i, nbad, stage, ply, caps_only, skip_quiets;   // skip_quiets: 1 = no more quiet moves, 2 = only those that give check
    int filt, ci_ok; U64 cs[6], dc;            // quiet list already cut down to checks; squares each piece type checks from, pieces that may uncover a check
    int pl[256], pol, pol_on, pol_best, pol_now, lm;   // lm: the scores come from the transformer (lm_get) instead; the read-out's score of each quiet move (x256), whether to use it at this node, the best and the current move's
    int hist, bad_now, see_now;                // about the move just returned: its history score (quiet moves), is it a losing capture, its exchange value
} Picker;
static inline void mp_init(Picker *p, Move ttm, int ply, int caps_only) {
    p->ttm = pseudo_legal(ttm) ? ttm : 0; p->stage = p->ttm ? ST_TT : ST_GENCAP; p->ply = ply; p->caps_only = caps_only;
    p->n = p->i = p->nbad = p->skip_quiets = 0; p->pol = p->pol_on = p->lm = 0; p->filt = p->ci_ok = 0;
#ifdef CHECK_PL
    { Move l[256]; int n = gen(l, 0), in = 0; for (int k = 0; k < n; k++) in |= l[k] == ttm; if (ttm && in != (p->ttm != 0)) { printf("pseudo_legal disagrees with the generator: move %d\n", ttm); fflush(stdout); abort(); } }
#endif
}
static inline int mp_check(Picker *p, Move m) {            // does this quiet move give check?
    if (!p->ci_ok) { check_info(p->cs, &p->dc); p->ci_ok = 1; }
    if (FLAG(m) == F_CASTLE) return 0;
    return (p->cs[P.sq[FROM(m)] % 6] & BIT(TO(m))) != 0 || ((p->dc & BIT(FROM(m))) && gives_check_quiet(m));
}
static inline void mp_filter(Picker *p) {                  // keep only the checking moves of the quiet moves not yet returned
    int k = p->i;
    for (int j = p->i; j < p->n; j++) if (mp_check(p, p->list[j])) { p->list[k] = p->list[j]; p->sc[k] = p->sc[j]; if (p->pol_on) p->pl[k] = p->pl[j]; k++; }
    p->n = k; p->filt = 1;
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
        if (p->skip_quiets == 1) { p->stage = ST_BAD; p->i = 0; break; }
        p->n = gen(p->list, 2); p->i = 0;
        if (p->skip_quiets == 2) mp_filter(p);
        if (p->pol && p->lm) {                          // the transformer's view of the quiet moves
            const LmEnt *e = p->n ? lm_get(p->list, p->n) : 0;
            if (e) {
                int lo = 0; for (int j = 0; j < e->n; j++) if (e->lp[j] < lo) lo = e->lp[j];
                p->pol_best = -1000000; p->pol_on = 1;
                for (int k = 0; k < p->n; k++) {
                    int key = FROM(p->list[k]) * 64 + TO(p->list[k]), v = lo - 256;     // a move it was not asked about counts as below all it was
                    for (int j = 0; j < e->n; j++) if (e->mv[j] == key) { v = e->lp[j]; break; }
                    p->pl[k] = v; if (v > p->pol_best) p->pol_best = v;
                }
            }
        } else if (p->pol) {                            // which quiet moves would a search choose here? (only at nodes with depth left for it to matter)
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
            if ((par[HIST_V2] & 2) && p->sc[k] < 800000000 && p->ply > 0 && move_stack[p->ply - 1] && m == cmove[piece_stack[p->ply - 1]][TO(move_stack[p->ply - 1])]) p->sc[k] = 700000000;
        }
        p->stage = ST_QUIET;
        /* fall through */
    case ST_QUIET:
        if (p->skip_quiets == 2 && !p->filt) mp_filter(p);
        while (p->skip_quiets != 1 && p->i < p->n) {
            int b = p->i;
            for (int j = b + 1; j < p->n; j++) if (p->sc[j] > p->sc[b]) b = j;
            Move m = p->list[b]; int v = p->sc[b]; p->list[b] = p->list[p->i]; p->sc[b] = p->sc[p->i];
            if (p->pol_on) { p->pol_now = p->pl[b]; p->pl[b] = p->pl[p->i]; }
            p->i++;
            if (m == p->ttm) continue;
            p->bad_now = 0; p->hist = v >= 700000000 ? hist_score(m, p->ply) : (p->pol_on ? hist_score(m, p->ply) : v);
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

// What is known about a late move before it is searched (the same record serves the collection build and the model below).
typedef struct {
    short hs, hmain, hc1, hc2, ev_alpha, v_alpha, caph, see, tt_alpha, best_alpha;
    unsigned char depth, legal, ply, rstd, nd, rootd, pmcount, nq, nc, piece, victim, prune, ttdepth, ttflag, from, to;
    unsigned char f1, f2; unsigned short gid;
} LRec;
static inline short clip16(int v) { return (short)(v > 32000 ? 32000 : (v < -32000 ? -32000 : v)); }
// The learned judgment of late moves (options Sel, SelA, SelB, SelFile): an additive model gives the log-odds that a full-depth
// look at the move would beat the bound; the late-move reduction is SelA/100 - SelB/100 x (log-odds + 5) plies. The model file
// is written by ext/lsel_fit.py from records of the collection build: a bias, then per feature its bucket edges and weights.
#define LS_NF 34
static const signed char ls_kind[LS_NF] = {0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 0, 1, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};   // 0 the value up to a cap, 1 edges, 2 edges and a bucket for "none", 3 one bit
static const signed char ls_cap[LS_NF] = {20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 5, 6, 3, 20};
static int ls_ne[LS_NF], ls_edge[LS_NF][20], ls_ok; static float ls_w[LS_NF][24], ls_bias;
static int ls_load(const char *path) {
    FILE *f = fopen(path, "r"); ls_ok = 0;
    if (!f) return 0;
    if (fscanf(f, "%f", &ls_bias) != 1) { fclose(f); return 0; }
    for (int k = 0; k < LS_NF; k++) {
        int nw; char bar[4];
        if (fscanf(f, "%d", &ls_ne[k]) != 1 || ls_ne[k] < 0 || ls_ne[k] > 20) { fclose(f); return 0; }
        for (int i = 0; i < ls_ne[k]; i++) if (fscanf(f, "%d", &ls_edge[k][i]) != 1) { fclose(f); return 0; }
        if (fscanf(f, "%3s %d", bar, &nw) != 2 || nw < 1 || nw > 24) { fclose(f); return 0; }
        for (int i = 0; i < 24; i++) ls_w[k][i] = 0;
        for (int i = 0; i < nw; i++) if (fscanf(f, "%f", &ls_w[k][i]) != 1) { fclose(f); return 0; }
    }
    fclose(f); ls_ok = 1; return 1;
}
static inline float ls_logit(const LRec *r) {
    const int v[LS_NF] = {r->depth, r->legal, r->ply, r->hs, r->hmain, r->hc1, r->hc2, r->ev_alpha, r->see, r->caph, r->tt_alpha, r->best_alpha, r->nq, r->nc, r->pmcount, r->piece, r->victim, r->ttflag, r->nd,
                          r->f1 & 1, (r->f1 >> 1) & 1, (r->f1 >> 2) & 1, (r->f1 >> 3) & 1, (r->f1 >> 4) & 1, (r->f1 >> 5) & 1, (r->f1 >> 6) & 1, (r->f1 >> 7) & 1,
                          r->f2 & 1, (r->f2 >> 1) & 1, (r->f2 >> 2) & 1, (r->f2 >> 3) & 1, (r->f2 >> 4) & 1, (r->f2 >> 5) & 1, (r->f2 >> 6) & 1};
    float z = ls_bias;
    for (int k = 0; k < LS_NF; k++) {
        int b;
        if (ls_kind[k] == 3) b = v[k];
        else if (ls_kind[k] == 0) b = v[k] > ls_cap[k] ? ls_cap[k] : v[k];
        else if (ls_kind[k] == 2 && v[k] == -32000) b = ls_ne[k] + 1;
        else { b = 0; while (b < ls_ne[k] && ls_edge[k][b] <= v[k]) b++; }
        z += ls_w[k][b];
    }
    return z;
}
#ifdef LSEL
// Collection build: a sample of the moves after the first one is searched at full depth whatever the rules say, and what
// was known about the move beforehand is written out together with what the full-depth look found (LSEL_FILE, LSEL_RATE).
static LRec *lsel_buf; static int lsel_n; static FILE *lsel_f; static unsigned lsel_rate = 64; static U64 lsel_ctr = 88172645463325252ULL; static unsigned short lsel_gid;
static void lsel_flush(void) { if (lsel_f && lsel_n) { fwrite(lsel_buf, sizeof(LRec), lsel_n, lsel_f); fflush(lsel_f); lsel_n = 0; } }
#endif
static int qsearch(int alpha, int beta, int ply) {
    if ((++nodes & check_mask) == 0) check_time();
    if (stop_flag) return 0;
    if (ply >= MAXPLY - 1) return evaluate();
    ST(0);
#ifdef SSTAT
    st_p[st_mode]++;
#endif
    TTE *e = tt_find(P.key);
    Move ttm = 0;
    if (e->key == P.key) {
        ttm = e->move;
        int s = e->score, fl = e->flag & 3;
        if (s > MATE_IN) s -= ply; else if (s < -MATE_IN) s += ply;
        if (fl == TT_EXACT || (fl == TT_LOWER && s >= beta) || (fl == TT_UPPER && s <= alpha)) { ST(25); return s; }
    }
    int chk = in_check(), best, stand = -INF, a0 = alpha;
    if (!chk) {
        corr_ply = ply; stand = best = eval_corr();
#ifdef SSTAT
        { int d_ = stand >= beta ? stand - beta : (stand <= alpha ? alpha - stand : -1); st_e[0][d_ < 0 ? 0 : (d_ < 50 ? 0 : (d_ < 100 ? 1 : (d_ < 150 ? 2 : (d_ < 200 ? 3 : (d_ < 300 ? 4 : 5)))))]++; }
#endif
        if (best >= beta) { ST(24); return best; }
        if (best > alpha) alpha = best;
    } else best = -INF;
    Move bestm = 0, m; int legal = 0;
    thr_stack[ply] = 0;
    if (ttm && !chk && !is_capture(ttm) && !PROMO(ttm)) ttm = 0;       // only captures are searched here
    Picker mp; mp_init(&mp, ttm, ply, !chk);                             // out of check: captures that do not lose material; in check: every move
    while ((m = mp_next(&mp))) {
        if (!chk) {
            int victim = FLAG(m) == F_EP ? PAWN : (P.sq[TO(m)] == EMPTY ? NO_PT : P.sq[TO(m)] % 6);
            if (!(par[OFF] & 1024) && !PROMO(m) && stand + mat_val[victim] + par[DELTA_Q] <= alpha) continue;   // delta pruning
            if ((par[SEARCH_V3] & 8) && !PROMO(m) && victim != NO_PT && stand + 120 <= alpha && see_val[victim] <= see_val[P.sq[FROM(m)] % 6] && see(m) <= 0) continue;
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
        int pvb = e->key == P.key ? (e->flag & 4) : 0;
        e->key = P.key; e->move = bestm; e->score = (short)s; e->depth = 0;
        e->flag = TTF(best >= beta ? TT_LOWER : (best > a0 ? TT_EXACT : TT_UPPER)) | pvb;
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
    ST(1); if (pv_node) ST(30);
#ifdef SSTAT
    st_d[depth > 23 ? 23 : depth]++; st_p[st_mode]++;
#endif

    TTE *e = tt_find(P.key);
    Move ttm = 0, excl = excluded[ply];
    int tt_score = 0, tt_depth = -1, tt_flag = TT_NONE, cutnode = cutn[ply], ttpv = pv_node;
    if (e->key == P.key) {
        ttm = e->move; tt_depth = e->depth; tt_flag = e->flag & 3; ttpv |= (e->flag >> 2) & 1;
        tt_score = e->score;
        if (tt_score > MATE_IN) tt_score -= ply; else if (tt_score < -MATE_IN) tt_score += ply;
        if (!pv_node && !excl && tt_depth >= depth &&
            (tt_flag == TT_EXACT || (tt_flag == TT_LOWER && tt_score >= beta) || (tt_flag == TT_UPPER && tt_score <= alpha))) {
            ST(2);
            if ((par[PRUNE_V2] & 4) && ttm && tt_score >= beta && P.sq[FROM(ttm)] != EMPTY && P.sq[FROM(ttm)] / 6 == P.side && !is_capture(ttm) && !PROMO(ttm)) {
                int bonus = par[HB_QUAD] * depth * depth + par[HB_LIN] * depth; if (bonus > par[HB_MAX]) bonus = par[HB_MAX];
                thr_stack[ply] = par[THR_HIST] ? attacks_by(P.side ^ 1) : 0; hist_update(ttm, ply, bonus);
            }
            return tt_score;
        }
    }
    thr_stack[ply] = par[THR_HIST] ? attacks_by(P.side ^ 1) : 0;
    if (par[LM_WAIT] == 2 && par[LM_DEPTH] && par[POL_W] && !chk && !excl && depth >= par[LM_DEPTH] && lm_ready()) lm_early();   // ask now, use at the quiet moves
#ifdef CHECK_THR
    { U64 t = attacks_by(P.side ^ 1); for (int q = 0; q < 64; q++) if (((t >> q) & 1) != (U64)attacked(q, P.side ^ 1)) { printf("attack map disagrees on square %d\n", q); fflush(stdout); abort(); } }
#endif
    corr_ply = ply;
    int ev = chk ? -INF : eval_corr(), raw_ev = ev;
#ifdef SSTAT
    if (!chk) { int d_ = ev >= beta ? ev - beta : (ev <= alpha ? alpha - ev : -1); st_e[1][d_ < 0 ? 0 : (d_ < 50 ? 0 : (d_ < 100 ? 1 : (d_ < 150 ? 2 : (d_ < 200 ? 3 : (d_ < 300 ? 4 : 5)))))]++; }
#endif
    evals_stack[ply] = ev;
    int improving = !chk && ply >= 2 && ev > evals_stack[ply - 2];
    int thr_node = (par[THR_LMR] && !chk && depth >= 3) ? cheap_threat() : 0;       // a cheaper enemy piece attacks one of the mover's: reduce its moves less
    if (!chk && tt_flag != TT_NONE && tt_score > -MATE_IN && tt_score < MATE_IN &&
        (tt_flag == TT_EXACT || (tt_flag == TT_LOWER && tt_score > ev) || (tt_flag == TT_UPPER && tt_score < ev))) ev = tt_score;
    int vm = 256, vcls = 0;                                                // pruning-margin multiplier (x256) from the reliability read-out; vcls: -1 reliable, +1 unreliable
    if (have_vol && !chk && ((par[VOL_MIX] && !pv_node && depth <= 8) || par[VOL_LMR] || par[VOL_NMP])) {
        int vol = eval_vol();
        if (par[VOL_MIX] && !pv_node && depth <= 8) { vm = 256 + par[VOL_MIX] * (vol * 256 / par[VOL_REF] - 256) / 100; vm = vm < 128 ? 128 : (vm > 512 ? 512 : vm); }
        vcls = vol * 10 <= par[VOL_REF] * 6 ? -1 : (vol * 10 >= par[VOL_REF] * 15 ? 1 : 0);
    }
    if (!pv_node && !chk && !excl && beta > -MATE_IN && beta < MATE_IN) {
        if (!(par[OFF] & 4) && depth <= 3 && ev + par[RAZOR] * depth * vm / 256 < alpha) {                     // razoring: hopeless at low depth -> resolve captures and leave
            int v = qsearch(alpha, beta, ply);
            if (v <= alpha) { ST(4); return v; }
        }
        if (!(par[OFF] & 2) && depth <= 8 && ev - par[RFP] * (depth - improving) * vm / 256 >= beta &&
            (!par[THR_RFP] || ev - par[RFP] * (depth - improving) * vm / 256 - par[THR_RFP] >= beta || !cheap_threat())) { ST(3); return ev; }   // reverse futility (a wider margin when a cheaper piece attacks one of ours)
        if (!(par[OFF] & 1) && allow_null && depth >= 3 && ev >= beta && !(par[NMP_TT] && tt_flag == TT_UPPER && tt_score < beta) &&
            (!par[NMP_MARGIN] || raw_ev >= beta + par[NMP_MARGIN] - par[NMP_MARGIN] * depth / 10) &&
            (P.bb[P.side][KNIGHT] | P.bb[P.side][BISHOP] | P.bb[P.side][ROOK] | P.bb[P.side][QUEEN])) {
            int R = par[NMP_BASE] + depth / par[NMP_DIV] + ((ev - beta) / par[NMP_EV] > 3 ? 3 : (ev - beta) / par[NMP_EV]);
            if (par[VOL_NMP]) R -= vcls;                                    // a reliable evaluation: trust the null move more; an unreliable one: less
            move_stack[ply] = 0; cutn[ply + 1] = !cutnode; ST(5);
            make_null();
            int v; STM(1, v = -search(depth - 1 - R, -beta, -beta + 1, ply + 1, 0));
            unmake_null();
            if (stop_flag) return 0;
            if (v >= beta) { ST(6); return v > MATE_IN ? beta : v; }
        }
    }
    if (ttm) ST(28);
    if (!(par[OFF] & 512) && depth >= 4 && !ttm && !excl) { depth--; ST(29); }                              // internal iterative reduction
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
                    if (e->key != P.key || e->depth < depth - 3) { e->key = P.key; e->move = m; e->score = (short)v; e->depth = (unsigned char)(depth - 3); e->flag = TTF(TT_LOWER) | (ttpv << 2); }
                    ST(7); return v;
                }
            }
        }
    }
    Move quiets[256], caps[64];
    int legal = 0, nq = 0, nc = 0, best = -INF, a0 = alpha;
    Move bestm = 0, m;
    U64 ls_thr = 0; int ls_thr_ok = 0;                                     // the opponent's attack map, worked out once per node when the judgment model needs it
    Picker mp; mp_init(&mp, ttm, ply, 0);
    mp.pol = have_pol && par[POL_W] && !chk && depth >= par[POL_DEPTH];
    if (par[LM_DEPTH] && par[POL_W] && !chk && depth >= par[LM_DEPTH] && lm_ready()) mp.pol = mp.lm = 1;
    int is_capture_tt = mp.ttm && is_capture(mp.ttm);
    while ((m = mp_next(&mp))) {
        if (m == excl) continue;
        int quiet = !is_capture(m) && !PROMO(m);
        int hs = quiet ? mp.hist : 0;
        int ext = 0;
        int samp = 0, prn = 0, want = 0; LRec lr;
#ifdef LSEL
        if (lsel_f && legal >= 1 && depth >= 2 && !excl) {
            unsigned rate = depth >= 8 ? lsel_rate / 8 : (depth >= 5 ? lsel_rate / 3 : lsel_rate); if (rate < 1) rate = 1;
            lsel_ctr = lsel_ctr * 6364136223846793005ULL + 1442695040888963407ULL; samp = ((lsel_ctr >> 33) % rate) == 0;
        }
#endif
        // singular extension: if every other move falls clearly short of the table move, search the table move deeper
        if (!(par[OFF] & 256) && m == ttm && !excl && depth >= par[SE_DEPTH] && tt_depth >= depth - 3 && (tt_flag == TT_LOWER || tt_flag == TT_EXACT) &&
            tt_score > -MATE_IN && tt_score < MATE_IN && ply + depth < MAXPLY - 8) {
            int sbeta = tt_score - par[SE_MUL] * depth; ST(19);
            excluded[ply] = m;
            int v; STM(2, v = search((depth - 1) / 2, sbeta - 1, sbeta, ply, 0));
            excluded[ply] = 0;
            if (stop_flag) return 0;
            if (v < sbeta) { ext = (par[SE_DOUBLE] && !pv_node && v < sbeta - par[SE_DOUBLE] && ply < 2 * root_depth) ? 2 : 1; ST(20); }
            else if (sbeta >= beta) { ST(21); return sbeta; }                        // several moves beat beta: cut
            else if (par[NEG_EXT] && (tt_score >= beta || cutnode)) ext = -1;   // not singular and probably a cutoff anyway: look less
        }
        if (!samp && (!pv_node || (par[PV_PRUNE] && ply > 0)) && !chk && best > -MATE_IN) {
            if (quiet) {
                int pd = depth;                                             // the depth the pruning rules reason about
                if (par[LMR_PRUNE]) { pd = depth - lmr[depth > 63 ? 63 : depth][legal + 1 > 63 ? 63 : legal + 1] - !improving; if (pd < 1) pd = 1; }
                if (par[PRUNE_V2] & 3) {
                    int rd = depth - lmr[depth > 63 ? 63 : depth][legal + 1 > 63 ? 63 : legal + 1] - !improving; if (rd < 1) rd = 1;
                    if ((par[PRUNE_V2] & 1) && rd < 5 && legal >= 1 && ply > 1 && move_stack[ply - 1] && move_stack[ply - 2] &&
                        cmh[piece_stack[ply - 1]][TO(move_stack[ply - 1])][(int)P.sq[FROM(m)]][TO(m)] < 0 &&
                        fuh[piece_stack[ply - 2]][TO(move_stack[ply - 2])][(int)P.sq[FROM(m)]][TO(m)] < 0) continue;
                    if ((par[PRUNE_V2] & 2) && rd <= 7 && legal >= 1 && ev + (par[FUT_BASE] + par[FUT_MUL] * rd) * vm / 256 + hs / 64 <= alpha) continue;
                }
                // both conditions stay true for every later quiet move of this node, so those are not even generated
                if (!(par[OFF] & 16) && depth <= par[LMP_DEPTH] && legal >= (par[LMP_BASE] + depth * depth) / (2 - improving)) { mp.skip_quiets = 1; ST(11); continue; }   // late-move pruning
                int gc = 0;
                if (par[SEARCH_V3] & 1) {                                   // quiet checks are judged like captures: only by the exchange
                    if (mp.skip_quiets == 2) gc = 1;                        // (the picker now hands out checks only)
                    else if ((pd <= 7 && ev + (par[FUT_BASE] + par[FUT_MUL] * pd) * vm / 256 <= alpha) || (pd <= 3 && hs < -par[HIST_PRUNE] * pd)) gc = mp_check(&mp, m);
                    if (gc) { if (depth <= 6 && see(m) < -par[SEE_C] * depth) { ST(14); continue; } }
                }
                if (!gc) {
                if (!(par[OFF] & 32) && pd <= 7 && ev + (par[FUT_BASE] + par[FUT_MUL] * pd) * vm / 256 <= alpha) { mp.skip_quiets = (par[SEARCH_V3] & 1) ? 2 : 1; ST(12); continue; }   // futility
                if (!(par[OFF] & 128) && pd <= 3 && hs < -par[HIST_PRUNE] * pd) { ST(13); continue; }                              // moves that keep failing
                if (!(par[OFF] & 64) && pd <= 6 && see(m) < -par[SEE_Q] * pd) { ST(14); continue; }                            // moves into a losing exchange
                }
            } else if (!(par[OFF] & 64) && depth <= 6 && mp.bad_now && mp.see_now < -par[SEE_C] * depth) { ST(15); continue; }
            else if ((par[SEARCH_V3] & 4) && !pv_node && legal >= 1 && P.sq[TO(m)] != EMPTY && !PROMO(m) && depth - lmr[depth > 63 ? 63 : depth][legal + 1 > 63 ? 63 : legal + 1] < 7 &&
                     ev + mat_val[P.sq[TO(m)] % 6] + 120 + 150 * (depth - lmr[depth > 63 ? 63 : depth][legal + 1 > 63 ? 63 : legal + 1] < 0 ? 0 : depth - lmr[depth > 63 ? 63 : depth][legal + 1 > 63 ? 63 : legal + 1]) <= alpha) continue;
            else if ((par[PRUNE_V2] & 16) && !pv_node && legal >= 1 && is_capture(m) && FLAG(m) != F_EP && !PROMO(m) &&
                     depth - lmr[depth > 63 ? 63 : depth][legal + 1 > 63 ? 63 : legal + 1] < 6 && caph[(int)P.sq[FROM(m)]][TO(m)][P.sq[TO(m)] % 6] < 0) continue;
        }
        want = samp || (par[SEL] && ls_ok && depth >= 3 && (quiet || par[CAP_LMR]) && legal + 1 > 1 + 2 * pv_node);
        if (want) {
            int f = FROM(m), t = TO(m), pc = (int)P.sq[f];
            if (!ls_thr_ok) { ls_thr = attacks_by(P.side ^ 1); ls_thr_ok = 1; }
            U64 thr = ls_thr;
            if (samp && (!pv_node || (par[PV_PRUNE] && ply > 0)) && !chk && best > -MATE_IN) {  // which rule would have dropped it
                if (quiet) {
                    if (depth <= par[LMP_DEPTH] && legal >= (par[LMP_BASE] + depth * depth) / (2 - improving)) prn = 1;
                    else if (depth <= 7 && ev + (par[FUT_BASE] + par[FUT_MUL] * depth) * vm / 256 <= alpha) prn = 2;
                    else if (depth <= 3 && hs < -par[HIST_PRUNE] * depth) prn = 3;
                    else if (depth <= 6 && see(m) < -par[SEE_Q] * depth) prn = 4;
                } else if (depth <= 6 && mp.bad_now && mp.see_now < -par[SEE_C] * depth) prn = 5;
            }
            memset(&lr, 0, sizeof lr);
            lr.hs = clip16(hs / 4); lr.hmain = clip16(quiet ? HIST(m, ply) / 2 : 0);
            lr.hc1 = (quiet && ply > 0 && move_stack[ply - 1]) ? cmh[piece_stack[ply - 1]][TO(move_stack[ply - 1])][pc][t] : 0;
            lr.hc2 = (quiet && ply > 1 && move_stack[ply - 2]) ? fuh[piece_stack[ply - 2]][TO(move_stack[ply - 2])][pc][t] : 0;
            lr.ev_alpha = chk ? -32000 : clip16(ev - alpha);
            lr.caph = (!quiet && P.sq[t] != EMPTY) ? caph[pc][t][P.sq[t] % 6] : 0;
            lr.see = clip16(see(m));
            lr.tt_alpha = tt_flag == TT_NONE ? -32000 : clip16(tt_score - alpha);
            lr.best_alpha = best <= -MATE_IN ? -32000 : clip16(best - alpha);
            lr.depth = (unsigned char)(depth > 255 ? 255 : depth); lr.legal = (unsigned char)(legal + 1 > 255 ? 255 : legal + 1); lr.ply = (unsigned char)ply;
            lr.rootd = (unsigned char)root_depth; lr.pmcount = (unsigned char)(ply > 0 ? (mcount[ply - 1] > 255 ? 255 : mcount[ply - 1]) : 0);
            lr.nq = (unsigned char)nq; lr.nc = (unsigned char)nc; lr.piece = (unsigned char)(pc % 6);
            lr.victim = (unsigned char)(FLAG(m) == F_EP ? PAWN : (P.sq[t] == EMPTY ? NO_PT : P.sq[t] % 6));
            lr.prune = (unsigned char)prn; lr.ttdepth = (unsigned char)(tt_depth < 0 ? 255 : tt_depth); lr.ttflag = (unsigned char)tt_flag; lr.from = (unsigned char)f; lr.to = (unsigned char)t;
            lr.f1 = (unsigned char)(pv_node | (cutnode << 1) | (improving << 2) | (chk << 3) | (quiet << 5) | ((m == killers[ply][0] || m == killers[ply][1]) << 6) | ((ttm != 0) << 7));
            lr.f2 = (unsigned char)((ttm && is_capture(ttm)) | ((!quiet && mp.bad_now) << 1) | ((PROMO(m) != 0) << 2) | (((thr >> f) & 1) << 3) | (((thr >> t) & 1) << 4) |
                                    (((pawn_att[P.side][t] & P.bb[P.side ^ 1][PAWN]) != 0) << 5) | ((tt_flag == TT_UPPER && tt_score <= alpha) << 6));
#ifdef LSEL
            lr.gid = lsel_gid;
#endif
        }
        move_stack[ply] = m; piece_stack[ply] = P.sq[FROM(m)];
#ifdef CHECK_GC
        int gcq = (quiet && FLAG(m) != F_CASTLE) ? mp_check(&mp, m) : -1;
#endif
        if (!make(m)) continue;
#ifdef CHECK_GC
        if (gcq >= 0 && gcq != in_check()) { printf("the check test disagrees with the board: move %d\n", m); fflush(stdout); abort(); }
#endif
        legal++; ST(26); mcount[ply] = legal;
#ifdef SSTAT
        st_dm[depth > 23 ? 23 : depth]++; if (quiet) st_dq[depth > 23 ? 23 : depth]++;
#endif
        int v, nd = depth - 1 + ext;
        if (legal == 1) { cutn[ply + 1] = pv_node ? 0 : !cutnode; v = -search(nd, -beta, -alpha, ply + 1, 1); }
        else {
            int r = 0;
            if (!(par[OFF] & 8) && depth >= 3 && (quiet || par[CAP_LMR]) && legal > 1 + 2 * pv_node) {
                r = lmr[depth > 63 ? 63 : depth][legal > 63 ? 63 : legal];
                if (!improving) r++;
                if (thr_node) r--;
                if (cutnode) r += par[CUT_LMR];
                if ((par[SEARCH_V3] & 2) && ttpv) r--;
                if ((par[PRUNE_V2] & 8) && ply > 0 && mcount[ply - 1] > 13) r--;
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
            if (want) {
                lr.nd = (unsigned char)(nd < 0 ? 0 : nd); lr.f1 |= (unsigned char)(in_check() << 4);
                if (par[SEL] && ls_ok && depth >= 3 && (quiet || par[CAP_LMR]) && legal > 1 + 2 * pv_node) {       // the learned judgment sets the reduction
                    float z = ls_logit(&lr);
                    r = (int)floorf((par[SEL_A] - par[SEL_B] * (z + 5.0f)) / 100.0f + 0.5f);
                    if (r < 0) r = 0;
                    if (r > nd - 1) r = nd - 1;
                }
                if (samp) { lr.rstd = (unsigned char)r; r = 0; }
            }
            cutn[ply + 1] = r ? 1 : !cutnode; if (r) { ST(16); STA(31, r); }
            if (r) { STM(5, v = -search(nd - r, -alpha - 1, -alpha, ply + 1, 1)); } else v = -search(nd - r, -alpha - 1, -alpha, ply + 1, 1);
#ifdef LSEL
            if (samp && !stop_flag) {                                       // what the full-depth look found
                lr.v_alpha = clip16(v - alpha); lr.f2 |= (unsigned char)((v > alpha) << 7);
                lsel_buf[lsel_n++] = lr; if (lsel_n >= 65536) lsel_flush();
            }
#endif
            if (v > alpha && r) { ST(17);
                // the shallow look liked the move. Much more than the best so far: verify one ply deeper; barely: the shallow answer will do
                int nd2 = nd;
                if (par[DEEPER]) nd2 += (v > best + 50 + 2 * nd) - (v < best + 10);
                cutn[ply + 1] = !cutnode;
                if (nd2 > nd - r) STM(3, v = -search(nd2, -alpha - 1, -alpha, ply + 1, 1));
            }
            if (v > alpha && v < beta) { cutn[ply + 1] = 0; ST(18); STM(4, v = -search(nd, -beta, -alpha, ply + 1, 1)); }
        }
        unmake();
        if (stop_flag) return 0;
        if (v > best) {
            best = v; bestm = m;
            if (v > alpha) {
                alpha = v;
                if (v >= beta) {
                    ST(8); if (legal == 1) ST(9); STA(10, legal); if (m == ttm) ST(27); if (quiet) ST(32);
                    if (ttm) { ST(33); if (legal == 1) ST(34); } else { ST(35); if (legal == 1) ST(36); if (!quiet) ST(37); else if (m == killers[ply][0] || m == killers[ply][1]) ST(38); STA(39, legal); }
                    if (par[HIST_V2] & 16) {                                 // the opponent's first-choice quiet move that led here has just been refuted
                        if (ply > 0 && move_stack[ply - 1] && mcount[ply - 1] == 1 && ust[hply - 1].cap == EMPTY && !PROMO(move_stack[ply - 1]) && FLAG(move_stack[ply - 1]) != F_EP) {
                            int bonus = par[HB_QUAD] * (depth + 1) * (depth + 1) + par[HB_LIN] * (depth + 1); if (bonus > par[HB_MAX]) bonus = par[HB_MAX];
                            cont_update(ply - 1, piece_stack[ply - 1], TO(move_stack[ply - 1]), -bonus);
                        }
                    }
                    if (quiet) {
                        if ((par[HIST_V2] & 2) && ply > 0 && move_stack[ply - 1]) cmove[piece_stack[ply - 1]][TO(move_stack[ply - 1])] = m;
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
    if (best < beta) { ST(22); STA(23, legal); }
    if (excl) return best;
    if ((par[HIST_V2] & 4) && bestm && best > a0 && best < beta && !is_capture(bestm) && !PROMO(bestm)) {      // raised alpha without a cutoff: still the move to try first next time
        int bonus = par[HB_QUAD] * depth * depth + par[HB_LIN] * depth; if (bonus > par[HB_MAX]) bonus = par[HB_MAX];
        hist_update(bestm, ply, bonus);
        for (int q = 0; q < nq; q++) if (quiets[q] != bestm) hist_update(quiets[q], ply, -bonus);
    }
    if ((par[HIST_V2] & 8) && best <= a0 && depth >= 3 && ply > 0 && move_stack[ply - 1] && ust[hply - 1].cap == EMPTY && !PROMO(move_stack[ply - 1]) && FLAG(move_stack[ply - 1]) != F_EP) {
        int bonus = par[HB_QUAD] * depth * depth + par[HB_LIN] * depth; if (bonus > par[HB_MAX]) bonus = par[HB_MAX];
        cont_update(ply - 1, piece_stack[ply - 1], TO(move_stack[ply - 1]), bonus * (1 + (pv_node || cutnode)));   // every reply failed: the move that led here was good
    }
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
        if (par[MM_CORR]) for (int k = 0; k < 2; k++) {
            c = &corr_mm[P.side][k][P.mmkey[k] & 16383];
            *c = (*c * (256 - w) + d * w) / 256;
            *c = *c > 12288 ? 12288 : (*c < -12288 ? -12288 : *c);
        }
    }
    int s = best;
    if (s > MATE_IN) s += ply; else if (s < -MATE_IN) s -= ply;
    e = tt_find(P.key);
    Move sm = bestm;
    if ((par[SEARCH_V3] & 48) && best <= a0) { Move old = e->key == P.key ? e->move : 0; sm = old ? old : ((par[SEARCH_V3] & 32) ? 0 : bestm); }
    if ((par[SEARCH_V3] & 64) && e->key == P.key && e->depth > depth + 3 && !(best > a0 && best < beta)) { if (best >= beta && bestm) e->move = bestm; e->flag |= ttpv << 2; }
    else {
    e->key = P.key; e->move = sm; e->score = (short)s; e->depth = (unsigned char)depth;
    e->flag = TTF(best >= beta ? TT_LOWER : (best > a0 ? TT_EXACT : TT_UPPER)) | (ttpv << 2);
    }
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
#ifdef LSEL
    lsel_gid++;
#endif
    memset(killers, 0, sizeof killers);
    corr_ply = 0; evals_stack[0] = in_check() ? -INF : eval_corr();
    root_vol = (have_vol && !in_check()) ? eval_vol() : 0;
    for (int i = 0; i < 2; i++) for (int t = 0; t < 4; t++) for (int a = 0; a < 64; a++) for (int b = 0; b < 64; b++) history_t[i][t][a][b] /= 2;
    thr_stack[0] = 0;
    int last = 0, stable = 0;
    tt_age = (tt_age + 1) & 31;
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
        memset(tt, 0, (tt_mask + 1) * sizeof(TTE)); memset(history_t, 0, sizeof history_t); memset(cmh, 0, sizeof cmh); memset(ch4, 0, sizeof ch4); memset(ch6, 0, sizeof ch6); memset(cmove, 0, sizeof cmove);
        memset(fuh, 0, sizeof fuh); memset(caph, 0, sizeof caph); memset(corrhist, 0, sizeof corrhist); memset(corr_mm, 0, sizeof corr_mm); memset(corr_np, 0, sizeof corr_np); memset(corr_cont, 0, sizeof corr_cont);
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
    init_nnz(); init_tables(); init_pext(); init_zobrist(); tt_alloc(32);
#ifdef SFNET
    if (getenv("SFNET_FILE") && !sf_load(getenv("SFNET_FILE"))) { printf("cannot read the network in SFNET_FILE\n"); return 1; }
#endif
    accs = (Acc *)big_alloc(sizeof(Acc) * 2048);
    init_lmr();
#ifdef LSEL
    if (getenv("LSEL_FILE")) { lsel_f = fopen(getenv("LSEL_FILE"), "ab"); lsel_buf = (LRec *)malloc(65536 * sizeof(LRec)); if (getenv("LSEL_RATE")) lsel_rate = (unsigned)atoi(getenv("LSEL_RATE")); }
#endif
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    if (argc > 1 && !strcmp(argv[1], "mob")) { eval_mode = 1; int unit[5] = {100, 100, 100, 100, 100}; set_piece_values(unit); }
    else if (argc > 1 && !load_net(argv[1])) {          // never fall back silently to the built-in evaluation: a wrong figure looks like a weak network
        fprintf(stderr, "cannot load network %s (missing, unknown format, or wider than this build's HMAX %d)\n", argv[1], HMAX); return 1; }
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
        else if (!strncmp(line, "setoption name SelFile value ", 29)) { if (!ls_load(line + 29)) printf("info string cannot read the model %s\n", line + 29); }
        else if (!strncmp(line, "setoption name TmNew value ", 27)) tm_new = atoi(line + 27);
        else if (!strncmp(line, "setoption name TmSoft value ", 28)) tm_soft = atoi(line + 28);
        else if (!strncmp(line, "setoption name ", 15) && strstr(line, " value ") && strncmp(line + 15, "Bonus", 5) && strncmp(line + 15, "EvalFile", 8)) {
            char *v = strstr(line, " value "); int len = (int)(v - (line + 15));
            for (int i = 0; i < NPAR; i++) if ((int)strlen(par_name[i]) == len && !strncmp(line + 15, par_name[i], len)) { par[i] = atoi(v + 7); if (par[i] < 1 && i != LMP_BASE && i < PROBCUT) par[i] = 1; }
            init_lmr();
        }
        else if (!strcmp(line, "ucinewgame")) { memset(tt, 0, (tt_mask + 1) * sizeof(TTE)); memset(history_t, 0, sizeof history_t); memset(cmh, 0, sizeof cmh); memset(ch4, 0, sizeof ch4); memset(ch6, 0, sizeof ch6); memset(cmove, 0, sizeof cmove); memset(fuh, 0, sizeof fuh); memset(caph, 0, sizeof caph); memset(corrhist, 0, sizeof corrhist); memset(corr_mm, 0, sizeof corr_mm); memset(corr_np, 0, sizeof corr_np); memset(corr_cont, 0, sizeof corr_cont); }
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
            check_mask = node_limit && node_limit < 200000 ? 0 : 2047;      // a small node budget is kept exactly (it used to overshoot to the next 2,048)
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
#ifdef TSTAT
            ts_start = __rdtsc();
#endif
            Move m = search_root(depth);
#ifdef TSTAT
            ts_go += __rdtsc() - ts_start; ts_nodes += nodes;
#endif
            if (print_root) {
                printf("info string rootscores");
                for (int i = 0; i < n_root; i++) { char ms[8]; move_str(root_moves[i], ms); printf(" %s %d", ms, root_scores[i]); }
                printf("\n");
            }
            use_bonus = 0;
            char ms[8];
            if (lm_fd >= 0) printf("info string transformer requests %llu, answers used %llu, answers too late %llu, waits %llu, waited %llu ms\n", lm_calls, lm_hits, lm_late, lm_waits, lm_wait_us / 1000);
            if (m) { move_str(m, ms); printf("bestmove %s\n", ms); } else printf("bestmove 0000\n");
        } else if (!strncmp(line, "perft", 5)) { printf("perft %llu\n", perft(atoi(line + 6))); }
        else if (!strcmp(line, "eval")) printf("eval %d\n", evaluate());
        else if (!strcmp(line, "fen")) { char f_[128]; pos_fen(f_); printf("fen %s\n", f_); }       // the position as the transformer is sent it
        else if (!strcmp(line, "qeval")) { nodes = 0; stop_flag = 0; node_limit = 0; hard_ms_limit = time_ms_limit = 0; printf("eval %d\n", qsearch(-INF, INF, 0)); }   // the value after captures are played out, as a search leaf returns it
        else if (!strcmp(line, "pol")) {                 // the read-out's scores for the quiet moves of the current position
            if (have_pol && !in_check()) { evaluate(); Picker q; mp_init(&q, 0, 0, 0); q.pol = 1; Move m; char ms[8]; while ((m = mp_next(&q))) if (q.pol_on && !is_capture(m) && !PROMO(m)) { move_str(m, ms); printf("pol %s %d\n", ms, q.pol_now); } }
            printf("pol end\n");
        }
        else if (!strcmp(line, "vol")) { evaluate(); printf("vol %d\n", have_vol ? eval_vol() : -1); }
#ifdef SFNET
        else if (!strcmp(line, "sfraw")) printf("sfraw %d\n", sf_ok ? sf_raw(0) : 0);
#endif
        else if (!strncmp(line, "see ", 4)) { Move m = parse_move(line + 4); printf("see %d\n", m ? see(m) : -99999); }
        else if (!strcmp(line, "quit")) {
#ifdef LSEL
            lsel_flush();
#endif
#ifdef TSTAT
            { unsigned long long tot = ts_go ? ts_go : 1; printf("info string evaluations %llu; cycles per evaluation: sums %.0f, read-out %.0f; share of all cycles: sums %.1f%%, read-out %.1f%%; cycles per position searched %.0f\n", ts_n, (double)ts_acc / (ts_n ? ts_n : 1), (double)ts_out / (ts_n ? ts_n : 1), 100.0 * ts_acc / tot, 100.0 * ts_out / tot, (double)ts_go / (ts_nodes ? ts_nodes : 1)); }
#endif
#ifdef SSTAT
            { U64 *t = st_; double all = (double)(t[0] + t[1]), cut = (double)t[8];
              printf("info string nodes %.0f: quiescence %.1f%% (stand-pat cut %.1f%% of them, table cut %.1f%%), main %.1f%% (PV %.2f%% of main)\n", all, 100 * t[0] / all, 100.0 * t[24] / t[0], 100.0 * t[25] / t[0], 100 * t[1] / all, 100.0 * t[30] / t[1]);
              printf("info string main nodes ended by: table %.1f%%, reverse futility %.1f%%, razoring %.1f%%, null move %.1f%% (tried %.1f%%, worked %.0f%% of tries), probcut %.2f%%, move loop cutoff %.1f%%, no cutoff %.1f%%\n",
                     100.0 * t[2] / t[1], 100.0 * t[3] / t[1], 100.0 * t[4] / t[1], 100.0 * t[6] / t[1], 100.0 * t[5] / t[1], 100.0 * t[6] / (t[5] ? t[5] : 1), 100.0 * t[7] / t[1], 100 * cut / t[1], 100.0 * t[22] / t[1]);
              printf("info string move-loop cutoffs: on the first move %.1f%%, by the table move %.1f%%, by a quiet move %.1f%%, mean move number %.2f; nodes with a table move %.1f%% of those reaching the loop; no-cutoff nodes searched %.1f moves on average\n",
                     100.0 * t[9] / cut, 100.0 * t[27] / cut, 100.0 * t[32] / cut, t[10] / cut, 100.0 * t[28] / (double)(t[8] + t[22]), (double)t[23] / (t[22] ? t[22] : 1));
              printf("info string cutoffs at nodes with a table move %.1f%% of all (on the first move %.1f%%); without one: on the first move %.1f%%, by a capture %.1f%%, by a killer %.1f%%, mean move number %.2f\n",
                     100.0 * t[33] / cut, 100.0 * t[34] / (t[33] ? t[33] : 1), 100.0 * t[36] / (t[35] ? t[35] : 1), 100.0 * t[37] / (t[35] ? t[35] : 1), 100.0 * t[38] / (t[35] ? t[35] : 1), (double)t[39] / (t[35] ? t[35] : 1));
              for (int k_ = 0; k_ < 2; k_++) { double te = 0; for (int i_ = 0; i_ < 6; i_++) te += (double)st_e[k_][i_]; printf("info string %s evaluations %.0f: within 50 of the window or inside %.1f%%, outside by 50-100 %.1f%%, 100-150 %.1f%%, 150-200 %.1f%%, 200-300 %.1f%%, 300+ %.1f%%\n", k_ ? "main-search" : "quiescence", te, 100 * st_e[k_][0] / te, 100 * st_e[k_][1] / te, 100 * st_e[k_][2] / te, 100 * st_e[k_][3] / te, 100 * st_e[k_][4] / te, 100 * st_e[k_][5] / te); }
              printf("info string nodes by purpose: first look %.1f%%, inside a null-move try %.1f%%, inside a singular test %.1f%%, inside a reduced look %.1f%%, re-search after a reduced look %.1f%%, full-window re-search %.1f%%\n", 100.0 * st_p[0] / all, 100.0 * st_p[1] / all, 100.0 * st_p[2] / all, 100.0 * st_p[5] / all, 100.0 * st_p[3] / all, 100.0 * st_p[4] / all);
              printf("info string main nodes by depth left (share of all nodes %%, moves searched per node, quiet share):");
              for (int d = 1; d < 24; d++) if (st_d[d]) printf(" %d: %.1f%% %.2f %.0f%% |", d, 100.0 * st_d[d] / all, (double)st_dm[d] / st_d[d], 100.0 * st_dq[d] / (st_dm[d] ? st_dm[d] : 1));
              printf("\n");
              printf("info string moves searched %.0f; skipped: late-move %llu, futility %llu, history %llu, losing quiet %llu, losing capture %llu (events; the first two end all quiets of the node)\n", (double)t[26], t[11], t[12], t[13], t[14], t[15]);
              printf("info string reduced searches %.1f%% of moves searched, mean reduction %.2f, re-searched %.1f%% of them; full-window re-searches %llu; singular tests %llu (extended %.0f%%, multi-cut %.0f%%); table-move-less depth cuts %llu\n",
                     100.0 * t[16] / t[26], (double)t[31] / (t[16] ? t[16] : 1), 100.0 * t[17] / (t[16] ? t[16] : 1), t[18], t[19], 100.0 * t[20] / (t[19] ? t[19] : 1), 100.0 * t[21] / (t[19] ? t[19] : 1), t[29]); }
#endif
#ifdef KSTAT
            printf("info string evaluations %llu, lazy view builds %llu, view rebuilds from the bucket cache %llu, pieces added or removed there %llu\n", ks_ens, ks_build, ks_ref, ks_diff);
#endif
            break;
        }
        fflush(stdout);
    }
    return 0;
}
