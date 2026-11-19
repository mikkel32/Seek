#!/bin/bash
cd /root/loopchess
while ! grep -q finished screen100k_b.log 2>/dev/null; do sleep 20; done
bash ext/q_fill.sh 5600 800
