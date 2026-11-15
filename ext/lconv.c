// Lichess evaluation database (one JSON object per line: a position and Stockfish evaluations of it) -> training shards.
//   curl -s https://database.lichess.org/lichess_db_eval.jsonl.zst | zstd -dc | ./lconv <out prefix> <shards>
//   python3 ext/pq2csv.py file.parquet | ./lconv <out prefix> <shards> csv      (the Hugging Face copy: one row per line of play,
//                                                                           rows of one position together: fen,first move,depth,cp,mate)
// Of each position the deepest evaluation's best line is taken. Kept: the side to move is not in check, the best move is
// not a capture or promotion, no forced mate (the same filter as our own self-play recorder). Written, for each kept
// position, to one of <shards> file pairs chosen at random (so every file is a random sample of the whole):
//   <prefix>_<k>.c   24 bytes: which squares are occupied (64 bits) + a 4-bit piece code per occupied square, in the
//                    mover's view (own P N B R Q K = 0-5, the other side's = 6-11; the board turned so the mover plays up)
//   <prefix>_<k>.y   1 byte: the mover's winning chance, 255 / (1 + exp(-centipawns / 271.58))
// THESE LABELS ARE STOCKFISH'S. A network trained on them is not self-taught.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
static int board[64];                     // 0 empty, 1-6 white P N B R Q K, 7-12 black
static int attacked_by(int s, int white) {            // is square s attacked by that colour?
    int r = s >> 3, f = s & 7, o = white ? 0 : 6;
    int pr = white ? r - 1 : r + 1;                   // a pawn of that colour stands one rank behind (from its view) and one file aside
    if (pr >= 0 && pr < 8) { if (f > 0 && board[pr * 8 + f - 1] == o + 1) return 1; if (f < 7 && board[pr * 8 + f + 1] == o + 1) return 1; }
    static const int nd[8][2] = {{1,2},{2,1},{-1,2},{-2,1},{1,-2},{2,-1},{-1,-2},{-2,-1}};
    for (int i = 0; i < 8; i++) { int rr = r + nd[i][0], ff = f + nd[i][1]; if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8 && board[rr * 8 + ff] == o + 2) return 1; }
    for (int dr = -1; dr <= 1; dr++) for (int df = -1; df <= 1; df++) {
        if (!dr && !df) continue;
        int rr = r + dr, ff = f + df;
        if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8 && board[rr * 8 + ff] == o + 6) return 1;
        int diag = dr && df;
        for (; rr >= 0 && rr < 8 && ff >= 0 && ff < 8; rr += dr, ff += df) {
            int p = board[rr * 8 + ff];
            if (!p) continue;
            if (p == o + 5 || p == (diag ? o + 3 : o + 4)) return 1;
            break;
        }
    }
    return 0;
}
static FILE **fc, **fy; static int K; static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static long long n = 0, kept = 0, s_check = 0, s_cap = 0, s_mate = 0, s_bad = 0;
// one position with its chosen evaluation: fen text (up to the first space the board, then the side to move), mate or centipawns
// (White's view), and the best move as text (4 or 5 characters)
static void process(const char *p, int is_mate, int cp, const char *ln) {
    n++; memset(board, 0, sizeof board);
    int r = 7, f = 0, ok = 1, wk = -1, bk = -1, np = 0;
    for (; *p && *p != ' '; p++) {
        if (*p == '/') { r--; f = 0; }
        else if (*p >= '1' && *p <= '8') f += *p - '0';
        else {
            const char *pc = "PNBRQKpnbrqk", *q = strchr(pc, *p);
            if (!q || r < 0 || f > 7) { ok = 0; break; }
            int code = (int)(q - pc) + 1; board[r * 8 + f] = code; np++;
            if (code == 6) wk = r * 8 + f; else if (code == 12) bk = r * 8 + f;
            f++;
        }
    }
    if (!ok || *p != ' ' || wk < 0 || bk < 0 || np > 32) { s_bad++; return; }
    int white = p[1] == 'w';
    if (is_mate) { s_mate++; return; }
    if (ln[0] < 'a' || ln[0] > 'h' || ln[2] < 'a' || ln[2] > 'h' || ln[1] < '1' || ln[1] > '8' || ln[3] < '1' || ln[3] > '8') { s_bad++; return; }
    int from = (ln[1] - '1') * 8 + (ln[0] - 'a'), to = (ln[3] - '1') * 8 + (ln[2] - 'a');
    int promo = ln[4] >= 'a' && ln[4] <= 'z';
    int pawn = board[from] == (white ? 1 : 7), enemy = board[to] && ((board[to] <= 6) != white);          // castling is written king-takes-own-rook: not a capture
    int capture = enemy || (pawn && (from & 7) != (to & 7));                                               // a pawn changing file onto an empty square: en passant
    if (attacked_by(white ? wk : bk, !white)) { s_check++; return; }
    if (capture || promo) { s_cap++; return; }
    if (!white) cp = -cp;
    if (cp > 3000) cp = 3000; else if (cp < -3000) cp = -3000;
    unsigned char rec[24]; memset(rec, 0, 24); uint64_t occ = 0; int k = 0;                               // the 24-byte record in the mover's view
    for (int s = 0; s < 64; s++) {
        int b = white ? s : s ^ 56, pcode = board[b];
        if (!pcode) continue;
        int mine = white ? pcode <= 6 : pcode >= 7, type = (pcode - 1) % 6, code = mine ? type : type + 6;
        occ |= 1ULL << s; rec[8 + k / 2] |= (unsigned char)(code << (4 * (k & 1))); k++;
    }
    memcpy(rec, &occ, 8);
    unsigned char y = (unsigned char)lrint(255.0 / (1.0 + exp(-cp / 271.58)));
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    int sh = (int)((rng >> 20) % (uint64_t)K);
    fwrite(rec, 1, 24, fc[sh]); fputc(y, fy[sh]); kept++;
}
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: lconv <out prefix> <shards> [csv]\n"); return 1; }
    K = atoi(argv[2]); fc = malloc(sizeof(FILE *) * K); fy = malloc(sizeof(FILE *) * K); char name[512]; int csv = argc > 3 && !strcmp(argv[3], "csv");
    for (int k = 0; k < K; k++) { snprintf(name, sizeof name, "%s_%d.c", argv[1], k); fc[k] = fopen(name, "wb"); snprintf(name, sizeof name, "%s_%d.y", argv[1], k); fy[k] = fopen(name, "wb"); if (!fc[k] || !fy[k]) { perror(name); return 1; } }
    size_t cap = 1 << 22; char *line = malloc(cap);
    if (csv) {                             // rows fen,first move,depth,cp,mate; a position's rows are consecutive: keep the first row of its greatest depth
        char cur[128] = "", bmv[8] = ""; int bdepth = -1, bcp = 0, bmate = 0;
        while (getline(&line, &cap, stdin) > 0) {
            char *c1 = strchr(line, ','); if (!c1) continue;
            char *c2 = strchr(c1 + 1, ','); if (!c2) continue;
            char *c3 = strchr(c2 + 1, ','); if (!c3) continue;
            char *c4 = strchr(c3 + 1, ','); if (!c4) continue;
            size_t fl = (size_t)(c1 - line); if (fl >= sizeof cur) continue;
            int depth = atoi(c2 + 1), same = strlen(cur) == fl && !memcmp(cur, line, fl);
            if (!same) {
                if (cur[0]) process(cur, bmate, bcp, bmv);
                memcpy(cur, line, fl); cur[fl] = 0; bdepth = -1;
            }
            if (depth > bdepth) {
                bdepth = depth; bmate = c4[1] != '\n' && c4[1] != '\r' && c4[1] != 0; bcp = atoi(c3 + 1);
                size_t ml = (size_t)(c2 - c1 - 1); if (ml > 5) ml = 5; memset(bmv, ' ', 6); memcpy(bmv, c1 + 1, ml); bmv[6] = 0;
            }
        }
        if (cur[0]) process(cur, bmate, bcp, bmv);
    } else while (getline(&line, &cap, stdin) > 0) {
        char *p = strstr(line, "\"fen\":\"");
        if (!p) { n++; s_bad++; continue; }
        p += 7;
        // the deepest evaluation: each one is ...,"knodes":N,"depth":D} after its lines; remember where its first line starts
        char *best = NULL; int bd = -1; char *e = p;
        while ((e = strstr(e, "\"pvs\":[{")) != NULL) {
            char *first = e + 8, *d = strstr(first, "\"depth\":");
            if (!d) break;
            int depth = atoi(d + 8);
            if (depth > bd) { bd = depth; best = first; }
            e = d;
        }
        if (!best) { n++; s_bad++; continue; }
        int is_mate = !strncmp(best, "\"mate\"", 6), cp = 0; char mv[8] = "      ";
        if (!is_mate) {
            if (strncmp(best, "\"cp\":", 5)) { n++; s_bad++; continue; }
            cp = atoi(best + 5); char *ln = strstr(best, "\"line\":\"");
            if (!ln) { n++; s_bad++; continue; }
            for (int i = 0; i < 5 && ln[8 + i] && ln[8 + i] != '"'; i++) mv[i] = ln[8 + i];
        }
        process(p, is_mate, cp, mv);
    }
    for (int k = 0; k < K; k++) { fclose(fc[k]); fclose(fy[k]); }
    fprintf(stderr, "%lld positions read; kept %lld (%.1f%%); left out: in check %lld, best move a capture or promotion %lld, forced mate %lld, unreadable %lld\n",
            n, kept, 100.0 * kept / (n ? n : 1), s_check, s_cap, s_mate, s_bad);
    return 0;
}
