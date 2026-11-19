#!/bin/bash
# keeps the OURO trainer alive: if it dies (for example because another GPU job crashed the shared GPU service), start it again.
# Stop for good with: touch zero/OURO_HALT zero/OURO_STOP
cd /root/loopchess
while [ ! -f zero/OURO_HALT ]; do
  rm -f zero/OURO_STOP
  nice -n -10 python3 zero/ouro_gpu.py 2>&1 | tee -a zero/ouro_stdout.log
  sleep 5
done
