#!/bin/bash
# Which search settings should the standing test use? The self-tuned ones (tuned at 5,000 nodes) against the hand-set ones,
# same network, at clock speed. The winner is written to zero/standing_engine.
cd /root/loopchess; L=zero/config_test.log; N=zero/test_net.bin
for tc in "10+0.1 200" "60+0.6 50"; do
  set -- $tc
  python3 gauntlet.py --us "calc/calc_t6 $N" --opp "hand-set search|calc/calc_untuned $N|Hash=64" --tc $1 --openings $2 --workers 48 --out zero/config_test.jsonl | tail -1 | cut -c1-300 >> $L
done
python3 - <<'PY' >> zero/config_test.log
import json
w = d = l = 0
for line in open('zero/config_test.jsonl'):
    r = json.loads(line); w += r['w']; d += r['d']; l += r['l']
s = (w + 0.5 * d) / (w + d + l)
pick = 'calc/calc_t6' if s > 0.5 else 'calc/calc_untuned'
open('zero/standing_engine', 'w').write(pick)
print(f'self-tuned against hand-set at clock speed, all games: {w}-{d}-{l}, score {s:.3f} -> the standing test uses {pick}')
PY
echo finished >> $L
