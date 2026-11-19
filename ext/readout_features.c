// Offline feature extractor. Use the production quantization, king buckets and evaluation.
// Build with gcc -O3 -march=native -DHMAX=768 ext/readout_features.c -lm -o readout_features.
#define main calc_uci_main
#include "../calc/calc_s19.c"
#undef main

#include "settled_leaf.h"

int main(int argc, char **argv) {
    int use_leaf = argc == 3 && !strcmp(argv[2], "--settle");
    if (argc != 2 && !use_leaf) return 1;
    init_nnz(); init_tables(); init_pext(); init_zobrist(); tt_alloc(1);
    accs = big_alloc(sizeof(Acc) * 2048);
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    if (!load_net(argv[1]) || !NH || NM != 1 || (NBK != 1 && NBK != 8)) {
        fprintf(stderr, "Expected a direct read-out with one or eight piece-count buckets.\n");
        return 1;
    }
    // Each binary row is bucket, baseline centipawns, and both views' squared activations.
    char fen[256];
    float row[2 + 2 * HMAX];
    while (fgets(fen, sizeof fen, stdin)) {
        set_fen(fen);
        int sign = 1, constant = 0, score = 0;
        if (use_leaf) {
            score = settle(0, -INF, INF);
            for (int i = 0; i < leaf_length[0]; i++) {
                if (!make(leaf_line[0][i])) return 1;
                sign = -sign;
            }
            constant = abs(score) >= MATE_IN || is_draw();
        }
        row[0] = NBK == 1 ? 0 : phase_bucket();
        row[1] = sign * evaluate();
        if (constant) {
            row[0] = -1;    // terminal score has no gradient with respect to a network weight
            row[1] = score;
        }
        for (int view = 0; view < 2; view++) {
            const short *a = accs[hply].a[P.side ^ view];
            for (int i = 0; i < NH; i++) {
                unsigned x = a[i] < 0 ? 0 : (a[i] > QU_ONE[i] ? QU_ONE[i] : a[i]);
                x *= (unsigned short)QU_MUL[i];
                row[2 + view * NH + i] = sign * (float)((x * x) >> 16) / 16384.0f;
            }
        }
        if (fwrite(row, sizeof(float), 2 + 2 * NH, stdout) != (size_t)(2 + 2 * NH)) return 1;
    }
    return ferror(stdin) || fflush(stdout) != 0;
}
