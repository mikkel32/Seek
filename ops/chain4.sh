#!/bin/bash
cd /root/loopchess
while ! grep -q finished screen100k_b.log 2>/dev/null; do sleep 15; done
bash ext/q_screen.sh calc/calc_s19f 100000 screen100k_c.log SearchV3=2 SearchV3=4 SearchV3=8 SearchV3=16 SearchV3=48 SearchV3=64
