#!/bin/bash
# let the running PolW 8000 run finish, record it, then start the non-blocking tests
cd /root/loopchess
while [ ! -e data/q/runs/lmtL1p_8_8000_1_300000.json ]; do sleep 10; done
echo "300000 nodes, LmDepth 8, PolW 8000, PolLmr 1 (old link, waiting): $(python3 ext/quality.py cmp lmtL1p_8_8000_1_300000 bigm_300000 2>&1 | tail -1 | sed -e 's/^.*difference //' -e 's/; negative = the first gives away less//')" >> lm_tree.log
for p in $(pgrep -f "lm/serve.py ckpt/L1p.pt /tmp/ouro_tree_L1p.sock"); do kill $p; done; sleep 2
rm -f zero/Q_BUSY
bash lm/lmtree_test.sh L1p lm_tree.log "300000 8 2000 1 0" "300000 8 2000 1 1" "300000 6 2000 1 0" "300000 10 2000 1 0"
