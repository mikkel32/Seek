#!/bin/bash
# When the first king-bucketed network has played: if it beat the plain one by more than 15 Elo, train the same network
# on all 7.06 billion positions with the same number of training samples (2.48 passes) and play it against the first.
cd /root/loopchess
while ! grep -q "BIGK finished" bigk.log 2>/dev/null; do sleep 30; done
elo=$(grep "BIGK (768 wide, 16 king buckets) against BIGW" bigk.log | grep -o '"elo": [-0-9.]*' | awk '{print int($2)}')
if [ "${elo:-0}" -gt 15 ]; then bash ext/kb_run.sh BIGK2 2.48 16 768 7 BIGK; else echo "[$(date -u +%H:%M:%S)] king buckets did not clear +15 (${elo:-none}): nothing queued" >> bigk.log; fi
