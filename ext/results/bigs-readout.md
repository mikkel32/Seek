# BIGS output-layer and scale experiment

The production checkpoint copied from the running instance is the 37,807,184-byte, 768-wide OUR3 network with 16 king buckets, eight output buckets and SHA-256 `9b204a84cbb78acab433ce79fbefdc848e02c71df2a2966e73affb3a2f2fe2fa`. Using the same training and held-out opening groups as G101, the fitted readout moved held-out cross entropy from 0.653299 to 0.653007. The fitted production model had SHA-256 `b6c8cf0b7b656b819916ec28c82b1a02a6e72497ab3b59b45edb7810f9e7205c`.

Both screens used the same 160 opening pairs, 15,000 nodes per move, offset 0, pool size 640, seed 181126 and the same s19 executable. One challenger changed only the output weights. The other changed only UCI EvalScale from 100 to 72. The selection rule fixed before play advances the higher scorer to 150 fresh opening pairs at 10+0.1 (offset 640, seed 49017), provided it scores at least 50% at fixed nodes. Each confidence interval treats the two colors of an opening as one observation.

| Challenger versus standing BIGS | Control | W–D–L | Elo | Paired 95% interval |
|---|---|---:|---:|---:|
| Archived-game readout | 15,000 nodes | 73–111–136 | −69.3 | −100.1 to −39.5 |
| Original BIGS with EvalScale 72 | 15,000 nodes | 99–133–88 | +11.9 | −14.8 to +38.8 |
| Original BIGS with EvalScale 72 | 10 s + 0.1 s, fresh openings | 33–219–48 | −17.4 | −36.9 to +2.0 |

The favorable scale-only point estimate at fixed nodes reversed under the project clock. All 940 games in this first experiment finished with legal moves and no time losses. Keep the standing BIGS configuration.

## Combining descendants of BIGS

C1a and G1a started from the same BIGS training checkpoint. C1a continued on the teacher data, while G1a added self-play. G1b used a stronger dose of self-play and lost to BIGS in the repository's earlier match. Two blends tested whether the descendants' changes could be combined while keeping a single evaluation per position. The average uses 25% BIGS, 50% C1a, and 25% G1a. The contrast starts from BIGS, adds 75% of the change from BIGS to C1a, and subtracts 25% of the change from BIGS to G1b. The fixed piece-value trailer comes from BIGS in both cases.

| Challenger versus standing BIGS | Control | W–D–L | Elo | Paired 95% interval |
|---|---|---:|---:|---:|
| Average | 20,000 nodes | 105–192–103 | +1.7 | −23.3 to +26.8 |
| Contrast | 20,000 nodes | 108–178–114 | −5.2 | −29.8 to +19.3 |
| Previously trained C1a control | 20,000 nodes | 115–172–113 | +1.7 | −23.5 to +27.0 |

The three matches used the same 200 pairs (offset 1920, seed 75219, pool size 640). Average and C1a tied on game points. The implemented tie-break selected C1a for 200 fresh pairs under 10+0.1 (offset 1280, seed 73148, pool size 640).

```sh
python ext/blend_bigs.py average w_BIGS.bin w_C1a.bin w_G1a.bin /tmp/average.bin
python ext/blend_bigs.py contrast w_BIGS.bin w_C1a.bin w_G1b.bin /tmp/contrast.bin
```

SHA-256 for the exported average was `1675f1e7e8d8e96ed85721fc01bbe2dde2122561accc58fb51e4c7262c7b4798`. SHA-256 for the contrast was `298405c63599afc540676c11fe87cb308728de08701dc2168b63040b319a8eba`.
