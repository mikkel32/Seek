#!/bin/bash
cd /root/loopchess
while ! grep -q "^done:" lm_L1p.log 2>/dev/null; do sleep 15; done
bash lm/lm_test.sh L1p lm_duo.log "30000 8 40" "300000 8 40" "300000 16 80" "30000 16 80"
