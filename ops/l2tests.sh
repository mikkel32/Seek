#!/bin/bash
cd /root/loopchess
echo "[$(date -u +%H:%M:%S)] L2 static judgment" >> lm_L2_tests.log
QUIET_BY="calc/calc_s19h zero/w_BIGS.bin" OURO_GPU=1 python3 ext/brain_lm.py ckpt/L2.pt 2 2>&1 | tail -8 >> lm_L2_tests.log
bash lm/lm_test.sh L2 lm_duo.log "30000 16 80" "100000 8 40" "300000 8 40"
echo "[$(date -u +%H:%M:%S)] finished L2 tests" >> lm_L2_tests.log
