/* Offline input-weight sensitivities at the capture-search leaf. */
#define main calc_uci_main
#include "../calc/calc_s19.c"
#undef main
#include "settled_leaf.h"

int main(int argc, char **argv) {
    int scores_only = argc == 3 && !strcmp(argv[2], "--scores");
    if (argc != 2 && !scores_only) return 1;
    init_nnz(); init_tables(); init_pext(); init_zobrist(); tt_alloc(1);
    accs = big_alloc(sizeof(Acc) * 2048);
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    if (!load_net(argv[1]) || !NH || NM != 1 || NPROD || have_mlp ||
        (NBK != 1 && NBK != 8) || getenv("QUNIT") || getenv("QBITS")) return 1;

    int32_t header[] = {NH, NKB * 768, q_bits};
    if (fwrite(header, sizeof *header, 3, stdout) != 3) return 1;
    char fen[256];
    while (fgets(fen, sizeof fen, stdin)) {
        set_fen(fen);
        int32_t score = settle(0, -INF, INF);
        if (fwrite(&score, sizeof score, 1, stdout) != 1) return 1;
        if (scores_only) continue;
        int sign = 1;
        for (int ply = 0; ply < leaf_length[0]; ply++) {
            if (!make(leaf_line[0][ply])) return 1;
            sign = -sign;
        }
        acc_ensure();
        int differentiable = abs(score) < MATE_IN && !is_draw();
        int bucket = NBK == 1 ? 0 : phase_bucket();
        int32_t rows[2][32];
        float gradient[2][HMAX];
        memset(rows, -1, sizeof rows);
        for (int view = 0; view < 2; view++) {
            int side = P.side ^ view;
            int king = lsb(P.bb[side][KING]), count = 0;
            for (int sq = 0; sq < 64; sq++) {
                int piece = P.sq[sq];
                if (piece == EMPTY) continue;
                if (count >= 32) return 1;
                rows[view][count++] = feat_fx(accs[hply].pb, side, piece / 6, piece % 6, sq, king);
            }
            const short *readout = view == 0 ? NWOq[bucket] : NWPq[bucket];
            const short *hidden = accs[hply].a[side];
            for (int j = 0; j < NH; j++) {
                float x = hidden[j] / (float)QU_ONE[j];
                gradient[view][j] = x > 0 && x < 1 && differentiable
                    ? sign * 271.58f * 2 * x * (readout[j] / 2048.0f) : 0;
            }
        }
        if (fwrite(rows, sizeof rows, 1, stdout) != 1 ||
            fwrite(gradient[0], sizeof(float), NH, stdout) != (size_t)NH ||
            fwrite(gradient[1], sizeof(float), NH, stdout) != (size_t)NH) return 1;
    }
    return ferror(stdin) || fflush(stdout) != 0;
}
