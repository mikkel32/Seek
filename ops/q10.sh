#!/bin/bash
# after the fine-tune probe: the full run's own recipe at probe scale (two teachers), then a smaller teacher share
cd /root/loopchess
while pgrep -f "lab_probe[9].sh" > /dev/null; do sleep 20; done
while [ ! -s zero/w_BIGX.bin.ema.pt ]; do sleep 20; done
bash lab_probe9.sh "distill_bigx_sf_half|--lr 1.4e-3 --distill zero/w_BIGX.bin.ema.pt+sf:/root/ext/sf141_embedded.bin --dw 0.5|h:0-33,s:0-33|zero/p_mixc.bin|p_mixc" "distill_sf_30|--lr 1.4e-3 --distill sf:/root/ext/sf141_embedded.bin --dw 0.3|h:0-33,s:0-33|zero/p_mixc.bin|p_mixc"
