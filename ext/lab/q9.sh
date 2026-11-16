#!/bin/bash
# waits for the second Stockfish-taught probe, then fine-tunes BIGM with Stockfish's network as the teacher
cd /root/loopchess
while pgrep -f "lab_probe[8].sh" > /dev/null; do sleep 20; done
bash lab_probe9.sh "ft_sf_half|--resume zero/w_BIGM.bin.ema.pt --lr 3e-4 --distill sf:/root/ext/sf141_embedded.bin --dw 0.5|h:0-33,s:0-33|zero/w_BIGM.bin|BIGM"
