#!/bin/bash
# One search technique at a time against the proven engine, 400 games each at 40,000 nodes per move.
# A watchdog pauses these games while a timed rating ladder runs, so the ladder's ratings stay clean.
cd "$(dirname "$0")/.."
OUT=logs/abtests.jsonl; NET=${NET:-zero/g4.bin}; NODES=${NODES:-40000}
( while true; do if pgrep -f elo_ladder.py >/dev/null; then pkill -STOP -f "calc/calc_ab"; else pkill -CONT -f "calc/calc_ab"; fi; sleep 3; done ) &
WD=$!
for opt in ${TESTS:-ProbCut=180 NegExt=1 SeDouble=25 CutLmr=1 TtCapLmr=1 CutIir=1}; do
  grep -q "\"opt\": \"$opt\", \"nodes\": $NODES" $OUT 2>/dev/null && continue
  python3 match.py --a calc:$NET:$NODES:calc/calc_ab:$opt --b calc:$NET:$NODES:calc/calc_ab --openings 200 --workers 2 2>/dev/null | tail -1 | \
    python3 -c "import sys, json; d = json.loads(sys.stdin.read()); print(json.dumps({'opt': '$opt', 'nodes': $NODES, **{k: d[k] for k in ('n', 'w', 'd', 'l', 'score', 'elo', 'lo', 'hi')}}))" >> $OUT
done
kill $WD; pkill -CONT -f "calc/calc_ab"
cat $OUT
