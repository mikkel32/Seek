#!/bin/bash
cd /root/loopchess
while ! grep -q finished screen100k.log 2>/dev/null; do sleep 10; done
bash ext/q_screen.sh calc/calc_s19e 100000 screen100k_b.log NmpMargin=40 NmpMargin=80 NmpMargin=130 LmrDiv=190 LmrBase=108 Rfp=85 Rfp=110
