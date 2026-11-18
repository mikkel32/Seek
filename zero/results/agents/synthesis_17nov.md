# Synthesis of 48 research reports (17 November)

Written by the synthesizing agent from the reports in `reports_17nov.json`. Estimates are the agents', not measurements; see `ext/README.md` for what was then measured.

SYNTHESIS OF 48 REPORTS (ranked by expected Elo per machine-hour at 10+0.1, 1 thread, vs SF 14.1)

Overall read: no researcher found a strong, externally measured lever. Almost every proposal is a +0 to +10 Elo item, and many sit below what our instruments can resolve (quality SE ~0.3 units = ~12 Elo; 1,000 games = ~6.5 Elo SE, 12 Elo at 95%). The best returns are (a) pure-speed changes that leave the search bit-identical, (b) diagnostics that decide where the 120 Elo search gap lives, (c) a gate on the pending PolW/PolLmr result that charges transformer latency. Dropped as contradicted by facts or too thin: more reduction/pruning threshold fishing (fact 8), MoE, bigger or scaled transformer as a player, MCTS/PUCT root layers, variance heads/B*/conspiracy numbers, root-prior retuning (fact 5), transformer value at leaves or end-of-line (fact 6), more fast-net generations (fact 8), sibling-pair retraining (fact 7), finer quantization (fact 8), wider net (fact 7), more transformer data/ChessBench shards/L2 corpus refresh (value 0 at the clock), Syzygy as a gap-closer (symmetric gain ~+7 for both sides, 8h), pawn-key LMR-only retunes.

TOP 15

1. Store raw static eval in the TT; skip evaluate() on non-cutoff hits (+ reuse eval in singular re-search, optionally after null move)
 Mechanism: TTE is 16 B with no eval field (calc_s19.c:369). Repack (32-bit key check or 16-bit move) to hold a short raw eval; store raw eval (before eval_corr) at every store (qsearch ~1532, probcut 1638, main 1876-1882); on hit use it, re-apply correction. Singular excluded search reuses evals_stack[ply]. Search tree stays bit-identical, so correctness is checkable by bench node signature.
 Evidence: evaluator is 45-55% of cycles (acc update 26-40%, read-out 13-19%); 92% of nodes evaluate; measured same-key entry present at eval time in 9.5-9.7% of evals on bench (best.bin, depth 12-14, TT cleared per position; higher across a real game). Weak point: internal-node accumulator work is only deferred, so true saving is mostly read-out + leaf builds. SF stores eval (memory, source not read).
 Expected Elo: +3 to +10 (nps +4 to +12%; ~50-100 Elo per node doubling at the clock).
 Cheap test: 30-min counter of skippable evals; stop if <25% of eval calls skippable; else implement, confirm identical bench nodes, compare nps (ext/nodetime.py, 3+ runs, quiet box); match only if nps >= +3-5%.
 Hours: 3.5-4. Proposed by: sf-search-features, nps-speed, qsearch, small-team-lessons, eval-cache-tt-eval, budget-plan.

2. Hash-sensitivity bisect (19% vs 1.1% move flips) plus no-evict TT replacement
 Mechanism: (a) zero-collision control: 2 GB vs 4 GB at 300k nodes. If flips stay >3%, something other than collisions leaks (stale history/corrhist/killers, ttpv and SearchV3 bits, TT-cutoff path). (b) null perturbations (hash 32/64/128, seed) as floor. (c) ablate TT cutoffs at non-PV, singular, probcut, TtCapLmr, NmpTt, corrhist/corr_np/corr_mm/corr_cont one at a time with flip rate and quality unit. (d) replacement fixes: qsearch/depth-0 stores must not evict a different-key entry with depth>0 (calc_s19.c:1532-1537; main store always replaces at 1876-1882); depth/age-gated same-key overwrite; 4-way cluster later. Diagnose why TtBucket=2 way is +0.72 worse (replacement score depth-8*age, 5-bit age wrap, qsearch evicting the deep pair entry).
 Evidence: vault_stab.log: 19.0% flips for 32 vs 64 MB and 18.5% for 512 vs 64 MB (so not table pressure); bucket variant 1.3-1.9%; SF 1.1%; TtBucket +19 Elo at 1 MB / -9 to -10 at clock; SearchV3=64 -0.17. 300k nodes at 4M slots is ~5-7% load.
 Expected Elo: -3 to +25, central about +3; real value is finding a latent defect.
 Cheap test: quality.py at 300k, hash 32/64/128 and 2G/4G, flips % and mean loss per variant; ~2 min engine time per run on 24 threads.
 Hours: 3 (diagnostic 1.5, fix 1.5). Proposed by: tt-stability, sf-search-features, fishtest-elo, gap-diagnosis, uncertainty-search, qsearch, eval-cache-tt-eval, novel-wildcards, root-search, pruning-math, singular-ext, nullmove-probcut, small-team-lessons.

3. Queen-aware 32 read-out heads in a full-scale BIGS-recipe net (zero runtime cost)
 Mechanism: retrain 768 wide, 16 mirrored buckets, --buckets 32 --obk 2 (piece count x my queen x opp queen); optionally try 64 heads. Do not spend more on input buckets (7 ~ 16 ~ 32 in held-out loss).
 Evidence: +11 (0 to +22) at 30k nodes, 2,000 games, vs probe r0 (README ~605-611); held-out loss gain tiny (0.54415 to 0.54397); input buckets saturated (loss 0.54263/0.54250/0.54253). Confirmation at full scale not done.
 Expected Elo: +3 to +10 at the clock (maybe 0 given fact 4).
 Cheap test: 1.09B probe pair (about 20 min GPU each) then quality at 300k plus 2,000 games at 30k nodes; then 1,000 at 10+0.1.
 Hours: 6 (mostly GPU, little match machine time). Proposed by: eval-arch, king-buckets.

4. Search micro-bundle behind par bits (each separately A/B on quality at 100k+300k; accept bundle only at <= -0.3 unit)
 Mechanism: (a) update continuation histories (cmh/fuh/ch4/ch6) from the post-LMR re-search outcome, then re-sweep LmrHist; (b) capture-history statScore for late captures (replace constant r -= 1/2 at line 1770); (c) fix improving: evals_stack stores -INF after check so improving is spuriously true; fall back to ply-4; (d) PV nodes reduce from move 2 (line 1756) with PvLmrFrom parameter; (e) signed LMR floor -1; (f) small ProbCut: return beta+M from TT lower bound at depth>=depth-3 (M 150/190/230); (g) TT-bound stand-pat in qsearch; (h) fail-high dampening (best*depth+beta)/(depth+1) for non-capture cutoffs; (i) default-defaults warning: standing options need PvPrune=1 ChkExt=0 SeDouble=20 CutIir=1.
 Evidence: SF regression-test page: stats updates after LMR for captures +32.07 (2022-01-29 test), capture statScore +11.88 (2024), 'Refine improving-logic' +47 (2019, partly unverified), small ProbCut ~+4 (commit 57e06be), PV LMR tweaks +37/+27. Counter: 12 reduction variants all slightly worse (fact 8); LmrHist=0 changed nothing (-2.3 Elo, -16 to +11); our ProbCut=150 neutral. Fishtest FAQ: per-patch gains biased upward, 1-3 Elo each typical.
 Expected Elo: +3 to +10 combined, many items will be 0.
 Cheap test: quality.py 300k (SE 0.3), repeat with different hash sizes (see #8); then 1,000 games only for a bundle <= -0.5.
 Hours: 4-5 total (0.5-2 per item). Proposed by: lmr-formula, fishtest-elo, nullmove-probcut, qsearch, eval-cache-tt-eval, small-team-lessons, aspiration-pvs, singular-ext.

5. Rule-50 damping and draw handling (native eval has no shuffling damp; sf_eval copy does, calc_s19.c:883)
 Mechanism: v = v*(100-min(P.half,100))/100 in evaluate() before corrhist; +-1 draw-score jitter; require 3-fold for repetitions wholly before root (is_draw at 1109-1115 scores a single prior occurrence as draw, runs before mate check at 1543); cuckoo upcoming-repetition detection; extend insufficient-material list. 62% of our games are draws.
 Evidence: code reading only; cuckoo in other engines +2.5 (+-1.9) to +11 (self-reported); SF jitter PR 1770 (no Elo given). Quality set has few high-halfmove positions, so quality test is nearly blind.
 Expected Elo: +1 to +5 combined.
 Cheap test: count games/positions with halfmove >= 30 first; build a targeted 500-position set from game logs; then 1,000 games.
 Hours: 2-2.5. Proposed by: quantization-accuracy, endgame-tb, aspiration-pvs, draw-conversion, novel-wildcards.

6. Time-management audit: clock trajectory log, then TmSoft/divisor sweep and SPSA of clock-allocation constants
 Mechanism: log ms spent, depth, clock left, hard-cap hits for 20 games at 10+0.1. By arithmetic (unmeasured) the clock drains in ~25-30 moves and the remainder gets about 100 ms. Then expose constants at calc_s19.c:2197-2205 and 1981-1983 (t/20, inc*3/4, hard cap 3x and 2/5, stability slope 0.05, drop slope 0.004) to zero/spsa_clock.py; also decayed best-move-change counter and predicted-iteration-cost stop; only TmSoft was ever SPSA'd (61 vs default 55).
 Evidence: self-reported small-engine PRs (+15 at 10+0.1, +27 bundled); our TM already has node-fraction, stability, drop, volatility; Vol TM was +10 (-7 to +27) at 5+0.05.
 Expected Elo: +3 to +8.
 Cheap test: 1 h trajectory diagnostic (go/no-go: more than 15% of moves hit the hard cap or the clock sits near 1 s from move 30); then three to five 1,000-game runs at ~20-25 min each, or SPSA 3-4 h at 44 boards (SE of tuned-vs-standing ~6-7 Elo at 4,000 games).
 Hours: 3-6. Proposed by: time-mgmt, budget-plan, lm-critical-positions, uncertainty-search.

7. Game autopsy: log moves + info lines in a gauntlet.py copy, re-run the standing 1,000 games, grade every ply with a local Stockfish oracle
 Mechanism: repo keeps NO game records (one_game returns only score/plies/flag). Classify the 342 losses by phase, our clock, depth reached, blunder vs bleed, q_classes class, endgame; calibrate our score vs oracle; null-normalize with SF's own moves. Yields a loss-position test set (~500-1,500 blunder positions) for an attribution matrix (3M nodes, SF net swap, hash sizes, in-game state vs cold start).
 Evidence: absence of records verified in gauntlet.py/match.py; quality.py positions are not the ones we reach or lose in. Oracle should be stronger than the opponent (local /usr/games/stockfish is v16 here, not 14.1).
 Expected Elo: 0 direct; aims all later hours (guess +10 to +30 if one class explains >=40% of lost win-prob).
 Cheap test: 50-game dry run to verify replay legality; then 1,000-game rerun (~25 min), oracle 200-300k nodes on ~150k positions (~10 CPU-min on 22 threads).
 Hours: 2-3. Proposed by: draw-conversion (sole author; it is unique and high-value).

8. Raise instrument power before testing small changes (null-perturbation calibration, hash-size replicate averaging, clustering by game, holdout, time-adjusted nodes, pentanomial logging, adjudication)
 Mechanism: run no-op perturbations (Hash 32/62/66/68/72/80, nodes +-1%) pairwise to get the empirical noise floor; average each position's loss over K=6 hash sizes before pairing; bootstrap by game (up to 8 positions per game are treated as independent in cmp); never use --subset blind spot (positions outside the subset are scored as no difference, quality.py:233-248); convert speed ratio to node budget; log pentanomial pair counts; mutual adjudication for 20-30% more games.
 Evidence: all 6+ paired rows in vault_calib.log have the right sign (~40 Elo/unit) but Sel was 5% slower (~7 Elo) and invisible to equal-node tests; HistV2 +0.49 unit vs -18 Elo at clock; q_tune baseline may be optimised on the same set so the 12 slightly-worse variants may be winner's curse. 2.6-sigma bucket result may be noise.
 Expected Elo: 0 direct; may rescue 1-2 dead ends and prevents misranking of speed-changing candidates.
 Cheap test: 3 extra quality runs plus one local cmp edit; 1.5 h.
 Hours: 1.5-2. Proposed by: statistics, reproduce-check, pruning-math, small-team-lessons, tuning-spsa.

9. Equal-depth factorization and per-iteration trace comparison (tree size per depth vs quality per depth; node-to-solution distribution)
 Mechanism: quality.py with go depth D=10,12,14 for ours-with-SF-net vs SF 14.1 on ~2,000 positions (nodes, seldepth, loss); then per-iteration EBF and nodes-to-stable-best-move ratio, stratified by piece count/tactical/volatility. Splits the 2-3x into 'bigger tree' vs 'worse move per depth', tail vs uniform. Uses only UCI output of SF (no source).
 Evidence: README depth-at-equal-nodes 14.5 vs 16.1 at 300k (12 positions only) yet SF needs 1/3 of the nodes; no equal-depth quality exists.
 Expected Elo: 0 direct; steers work (maybe +10 to +40 if it names one mechanism).
 Cheap test: ~30-60 min machine time; parsing script <100 lines.
 Hours: 3. Proposed by: gap-diagnosis (also reproduce-check node-count audit, see Disagreements).

10. Charge-and-gate the transformer-in-tree (PolW/PolLmr); non-blocking client with virtual latency only if gate passes
 Mechanism: step 0 arithmetic: 250 calls x 7 ms = 1.75 s vs ~0.27 s search; one unit = 1.21x nodes = ~8 calls, so compare PolW at equal TIME (about 2.2M-node baseline), not equal nodes. Step 1: sweep LmDepth (10/8/6/5), PolW-only vs PolLmr-only, q-gap variant, with a fake zero-latency oracle. Step 2 (only if zero-latency gain >= ~0.5-1.0 unit at 300k): LmLat=L virtual latency (reply visible after nodes >= issue+L, L~8,000-10,000) in quality tests (accept if it keeps >=70% of LmLat=0 gain), then real non-blocking socket client with PENDING/READY cache states, persistent cross-move cache, polling every 1,024 nodes, cap ~64 in flight, speculative requests along the TT chain. Do not clear lm_cache between quality positions (contaminates results). Binary protocol + engine-side legal moves + CUDA graph server (target ~1-2 ms) is an enabler only after this passes.
 Evidence: clock result -122 vs -111 with ~1 call/move; root prior -2.5/-0.8/~0 units at 30k/100k/300k; fact 8 warns better predictors may not convert. Server latency estimates are unmeasured (profile first, 1 h).
 Expected Elo: 0 to +10 (central +3), bounded by fact 5; negative if calls are paid synchronously.
 Cheap test: as listed; match with <=8 concurrent games only after gate.
 Hours: 0.5 (arithmetic) + 4 (virtual latency) + 6-7 (real async/server) = ~5 to gate, ~12 full. Proposed by: hybrid-lit, async-gpu, fast-infer, policy-lmr, moe-latency, lm-in-tree, lm-in-tree-design (sic), lm-small-student, reproduce-check, budget-plan, lm-architecture, mcts-ab.

11. Decouple ordering score from the LMR/pruning history sum; pawn-structure history; cutoff statistics first
 Mechanism: hist_score() feeds ordering, LMR (line 1768) and pruning (1692-1715), so HistV2/ThrHist ordering gains at 10k nodes were confounded with shifted reduction/prune scales (+19 at 10k, -10/-9/-18 deeper). Add ord_score used only in ST_GENQUIET (sum main + cmh*2 + fuh + ch4 + ch6 + pawn history keyed pkey&511 + optional low-ply history); keep old hs for LMR/prune. First log cutoff stats by stage/depth/rank and an oracle 'wasted siblings' node share (30 lines) to bound headroom (first-move cutoff already 82%). Also retune bonus scale 5d^2+33d cap 1322 vs SF-like at 300k with rescaled thresholds; make ThrHist cheap (lazy thr_stack; 60% of nodes end by RFP before use).
 Evidence: calc_s19.c:1417 and README 154-167; ThrHist +16 at equal nodes, -9 (-21 to +3) at clock; pawn history exists in later SF (fishtest +10 for one refinement, original unverified). Measured evidence for gain here is indirect.
 Expected Elo: 0 to +10.
 Cheap test: 1.5 h stats; then quality at 100k/300k plus SF-net yardstick at 100k (SE 0.33-0.40, repeat 3x).
 Hours: 4-5. Proposed by: move-ordering, history-heuristics-math.

12. Correction-history rework: additive weighted sum, two-ply continuation key, per-side minor/major, depth-scaled gravity update, larger tables
 Mechanism: replace nested averaging (2*c+np0+np1)/4 and (3*c+cont)/4 (calc_s19.c:1136-1140) by corr=(wP*pawn+wNp*(npW+npB)+wMm*mm+wCont*cont)/256 with par weights; SPSA them (MmCorr already 173). NpCorr=0 gave +17 (-4 to +38) at 30k nodes; MmCorr=256 +14.9; ContCorr +5. Update EMA weight cap 16/256 is low; try gravity form and 65536-entry tables.
 Evidence: internal 30k-node results (may fade at 300k like the transformer prior); wiki lists variants but no Elo; Stockfish introduction ~13 Elo, small-engine PRs +7 to +10.
 Expected Elo: +3 to +10, could be 0.
 Cheap test: quality at 300k; 30k-nodes-per-move self-play lab (2,000 games ~2 min); then 1,000 at 10+0.1.
 Hours: 3-5. Proposed by: correction-history, tree-shape-ml, lm-value-in-search, pruning-math.

13. Speed under match load: shared mmap read-only net, PGO/LTO, AVX-512 VNNI read-out, THP check
 Mechanism: diagnostic first: bench with 1, 12, 22 concurrent processes; if per-process nps drops >25% beyond SMT sharing, cache/L3 pressure is real. Each process fread()s its own ~19 MB net (calc_s19.c:501-531) so 22 copies compete for L3; mmap MAP_SHARED/ /dev/shm. PGO (reported 3-7% in forums; one local test 0.99x), AVX-512BW+VNNI qdot/tdot (microbench -25 to -40% of read-out on the Intel-class sandbox; Zen 4 double-pumps so gain smaller, unmeasured), quiet-move SIMD max scan (~1-2%), Finny table refresh share (profile first; only if >5% of time).
 Evidence: BIGX lost 45% nps under 76 copies vs 26% quiet (README ~330-345); attempted prefetch/hand-AVX2/lazy-eval/PGO-on-VM all null in nps-speed's tests. Sandbox measurements are noisy shared-VM numbers with a random net, not BIGS.
 Expected Elo: +3 to +10 if diagnostic shows load degradation; +2 to +5 otherwise.
 Cheap test: 22-process nps bench old vs new (ext/nodetime.py); nps gains are deterministic, 1,000-game sanity match only.
 Hours: 0.5 diagnostic, 4-5 for changes. Proposed by: budget-plan, nps-speed, king-buckets, small-team-lessons.

14. Quiet-position distillation into the fast net (transformer value as third teacher), gated by a static test first
 Mechanism: label 5M-150M quiet positions with L2/L1p value (batched on the 4080, not via serve.py), fine-tune or probe BIGS with half labels + half mean(teachers) incl. transformer, matched control on the same positions. Gate on ext/brain_quiet.py quiet-pair accuracy (BIGS 73.2% to >=74.5%) before any engine/quality time. Check first that transformer value MSE vs SF 4M reference beats BIGS 22.7.
 Evidence: quiet pairs 75.6-76.0% vs 73.2% vs SF net 72.0%; teachers have helped (+38 with SF net); but fact 7 (pairs learned, no transfer), BIGX width no better, teacher added to finished net +7 only (-4 to +18). Transformer inference throughput unmeasured.
 Expected Elo: -5 to +15, central +3; kill if <+1.0 pt on quiet pairs.
 Cheap test: step 1 30 min on 5M held-out; step 2 2 h on 20M; quality 300k only if gate passes.
 Hours: 6-8. Proposed by: searchless-scaling, lm-data, budget-plan, training-data, lm-architecture, lm-small-student, lm-in-tree-design.

15. Attack-map/threat inputs, narrow slice (--am --amk 64-128), gated by static probe before any engine work
 Mechanism: train 1.09B-position probe with --am class --amk 64 (trainer supports it; engine does not) vs r0 (held-out 0.54415); score quiet pairs and the 10,334 clear cases. Go to engine only if quiet-pair >= +0.7-1.0 pt or held-out loss drops >= 0.001-0.003. Engine side: per-ply 12x64 attack-map array, ~12.2 flipped attack inputs/move vs 2.2 piece inputs. Companion diagnostic: --am plus --sib, to test if inputs, not signal, block the sibling-pair transfer (fact 7).
 Evidence: own probe: attack maps -0.0032 loss, narrow 128/384 keeps ~85% of the gain, but layered read-out similarly gained +12 Elo for -46% speed and loss gains overstate Elo; shelved at 59M positions, never retested at 8.2B. External: SF18 threat inputs ~+14 on ARM LTC, neutral on x86 initially, severe nps loss and STC regression at first; PlentyChess +45 at fixed nodes but +1.8 (+-1.4) at 40+0.4 and -3.1 at STC; all secondhand.
 Expected Elo: -5 to +8 net of an assumed 10-25% nps loss, central ~0-3. Sensible only as a 3 h static gate.
 Cheap test: probe training ~3 h GPU, no engine build.
 Hours: 3 (gate), +8-16 if it passes. Proposed by: lm-distill-relational, threat-inputs, eval-arch, king-buckets, lm-architecture.

Items just below the cut: SPSA of search parameters on the quality instrument at 300k (+0-8, 8h; try only after #8 and a sensitivity sweep, 4 h); learned null-move gate (+0-15, central +3, 6h; null tries are 17-20% of nodes, 41-49% succeed, unlike LMR not re-checked); SelPrune (learned move-judgment model as pruner, -10 to +15, 4h); distilled policy into POL1 read-out/NNUE head (-10 to +15, 7-10h; POL1 gained at fixed nodes (+13/+25) but lost -53 at the clock per zero/best_options.md); root RootLmr/RootOrd retest at 300k (-3 to +6, 1.5h); Syzygy symmetric both-sides (+2-8, 8h).

MEASUREMENT FLAWS REPORTED
- Fixed-node quality tests charge no latency or per-node time: fact 5/11 results (and PolW/PolLmr) do not transfer to the clock; also blind to nps changes (ThrHist, Sel -5% speed, TT eval).
- Standing -111 is against an SSE2-only Stockfish 14.1 build that is 1.77x slower per node than ours (README 84-87; 194k vs 351k nps), so we get almost 2x its nodes; real gap to a normal build is roughly 100 Elo larger (equal-node ladder shows -218 to -247). Relative comparisons stay valid.
- Node-count asymmetry: calc counts every search()/qsearch() call incl. null-move children and singular re-searches (lines 1487, 1547, 1608); SF counts moves made (from memory, source not read). The 2.0/2.8/3.1x gap may be overstated by ~5-25%.
- quality.py: --subset treats positions outside the subset as no-difference (233-248); up to 8 positions per game treated as independent; truth = SF 4M nodes flatters SF-like choices; positions come from 30k-node BIGM-vs-SF games but are judged at 300k-1M; baseline/options may be tuned on the same set (winner's curse); calibration ~40 Elo/unit rests on ~6 rows, mostly node/network changes.
- Fact 9 internally odd: 19% flips for table size at 64 MB with ~5% load vs 0.1% for a 1% node change; two-entry bucket +0.72 (0.28) is ~2.6 sigma, may be noise; 512 MB plays same as 64 MB.
- match.elo_from uses a trinomial SE although colours are paired (CIs conservative); gauntlet --stop with imap_unordered/pool.terminate discards in-flight (long drawn) games, biasing early-stopped Elo toward decisive results; same first-N openings (seed 7) reused; openings from ChessBench test split (Lichess-derived, possible train overlap); load average ~80 on 24 threads in some logs (wall-clock allocation tuned at different board counts).
- Fast-net fixed-node results: HistV2 +19 at 10k nodes, -10/-9/-18 deeper; all shallow tunings (2.5k-30k nodes) mis-rank history options.
- Sandbox caveats: BIGS/BIGM nets not in checkout; local stockfish is v16; researchers' nps and node-share numbers used a random or tiny net on a noisy shared VM; latency numbers for the transformer server (1-2 ms target, python-chess ~0.3 ms) are unmeasured estimates; no clock trajectory exists in the repo; no game records exist anywhere.
- Code details: calc_s19.c defaults differ from standing options (ChkExt=1, PvPrune=0, SeDouble=0, CutIir=0); cache lm_cache is 2^15 direct-mapped, 96-move cap penalises legal moves beyond 96, repetition history dropped from FEN; improving flag is spuriously true after a check at ply-2; is_draw runs before mate check.
- External Elo evidence is mostly self-reported small-engine PRs, forum posts and secondhand search snippets; fishtest pages and several TalkChess fetches returned 503/404; no per-patch Elo table could be retrieved.

INJECTED-INSTRUCTION AND RULE WARNINGS
- No researcher reported any prompt-injection text in fetched pages (several explicitly stated none seen; many fetched nothing).
- Rule concern: lm-distill-relational fetched a Hugging Face Space containing what appears to be GPL-3 Stockfish threat-feature source (full_threats.cpp, header 'The Stockfish developers'). Only its high-level description was used; treat that description as contaminated if clean-room matters and tell the user (the Stockfish-source fetch is a pending user decision).
- threat-inputs read Stockfish commit text and PR summary (not source) and listed licences from memory unverified (Stockfish/PlentyChess/Monty GPLv3; Reckless AGPL or MIT, pages disagree). Fathom MIT and ChessBench CC0/CC-BY licences stated from memory. Do not copy GPL code (CuckooChess, Ergodice lc0 fork).
- Several proposals suggest using the local or a Stockfish 14.1 binary over UCI as a labelling oracle (autopsy, relabelling, depth-factorization); this is not source reading but check it against the user's pending Stockfish decision.

THREE MOST IMPORTANT DISAGREEMENTS
1. Is the 19% hash-size move instability a defect or harmless chaos? tt-stability, sf-search-features, uncertainty-search, gap-diagnosis, root-search, small-team-lessons and novel-wildcards treat it as a likely bug or leak (rank diagnosing it first; gap-diagnosis notes 19% vs 0.1% is inconsistent unless the test used a small table); budget-plan, singular-ext and others say it is probably near-tie chaos, point to 512 MB = 64 MB playing the same and to two-entry buckets (1.3% flips) playing +0.72 worse, and advise against spending hours. Cheapest resolution is the 2 GB vs 4 GB control (proposal #2, 1.5 h).
2. Is any further investment in the transformer in-tree worthwhile? async-gpu, fast-infer, mcts-ab-lite design proposals, lm-in-tree-design and moe-latency propose building non-blocking clients, CUDA-graph or binary servers and distilled students (expected 0 to +15); hybrid-lit, searchless-scaling, lm-critical-positions, lm-generations, lm-small-student, lm-architecture and budget-plan say fact 5/6/8 show the signal is absorbed by 300k nodes and in-tree calls cost clock time, so gate everything on the pending PolW/PolLmr zero-latency result and stop if within +-0.3 unit. Consensus: do not build infrastructure before the gate; they disagree on whether to spend the ~5 h of virtual-latency/server work at all.
3. Should hours go to evaluator relational inputs (threat/attack maps, transformer distillation, aux heads) or to search? lm-distill-relational, threat-inputs, eval-arch, lm-architecture and lm-data propose threat/attack-map inputs and distillation, citing the quiet-pair gap (76.0 vs 73.2%) and SF18's threat inputs; searchless-scaling, king-buckets, nps-speed, sf-search-features and others argue the evaluator already ties SF 14.1 (fact 4), loss gains convert poorly (layered read-out +12 for -46% speed; N384H +10 for -46%), x86 threat gains were neutral/regressing at short TC, and nps cost eats the benefit at 1.2 Mnps, so effort should go to search efficiency. Net advice: only the 3 h static gates (#14, #15) are justified before search diagnostics (#2, #7, #9) return.
(Runner-up disagreement: whether SF-style search tweaks can still pay at all. lmr-formula, fishtest-elo and nullmove-probcut offer +2 to +10 from fishtest-measured items but all warn that 12 variants and SF parameter groups showed nothing, and that single-patch gains of 1-3 Elo are below quality SE 0.3 units.)
