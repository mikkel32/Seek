# The strongest known configuration (15 November)

Engine `calc/calc_s9.c`, network `zero/nets/g101_vol.bin` (generation 101 with the reliability read-out), UCI options:

    VolMix=150 VolRef=70 SeDouble=20 CutIir=1 VolLmr=1 ContCorr=1 ChkExt=0 PvPrune=1 VolTm=1

Measured (same network on both sides, every opening with both colours):

| Against | Time control | Games | Result |
|---|---|---|---|
| `calc_s4`-equivalent (the same binary with default options) | 10+0.1 | 180 | 51-101-28, +45 Elo (11 to 79) |
| old engine (`calc_s0.c`) vs `calc_s4.c` | 10+0.1 | 132 | 49-60-23, +69 (26 to 115) |

Found afterwards and not yet in the combined test: `NpCorr=0` (+17, −4 to 38, at equal positions per move).
The clock-based tuning of the numeric parameters (`zero/spsa_clock.py`) starts from this configuration.

`zero/nets/g102.bin` is generation 102 without read-outs; `zero/nets/g101_vol_pol.bin` adds the where-to-search read-out
(`PolW=1000` was +13, −6 to 32, at equal positions; heavier weights lose).
