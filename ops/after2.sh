#!/bin/bash
cd /root/loopchess
bash lm/lm_test.sh L1p lm_duo.log "100000 8 40" "100000 16 80"
bash duo_clock.sh L1p 8 40 500
