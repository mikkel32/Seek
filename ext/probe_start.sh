#!/bin/bash
cd /root/loopchess
while [ "$(grep -c fetched fetch.log 2>/dev/null)" -lt 2 ]; do sleep 10; done
C="--bs 32768 --lr 1.4e-3"
bash probe_q.sh "plain768|--h 768 $C" "kb16|--h 768 --kb 16 --kf 1 $C" "kb16_no_shared|--h 768 --kb 16 $C" "kb32|--h 768 --kb 32 --kf 1 $C" "kb7|--h 768 --kb 7 --kf 1 $C" \
  "kb16_lr0.7|--h 768 --kb 16 --kf 1 --bs 32768 --lr 0.7e-3" "kb16_lr2.8|--h 768 --kb 16 --kf 1 --bs 32768 --lr 2.8e-3" "kb16_bs16k|--h 768 --kb 16 --kf 1 --bs 16384 --lr 1e-3" \
  "kb16_w1024|--h 1024 --kb 16 --kf 1 $C" "kb16_products|--h 768 --kb 16 --kf 1 --prod 1 $C" "kb16_layers|--h 768 --kb 16 --kf 1 --head 16,32 --hk 64 $C"

