#!/bin/bash
# The standing against Stockfish 14.1 at 10+0.1 with one game per physical core (44 boards) instead of 76:
# does the usual load flatter or hurt us against Stockfish (its network is larger than BIGM's)?
cd /root/loopchess
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; SF="Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1"
flock /dev/shm/cpu.lock bash gt.sh "standing, BIGM: against Stockfish 14.1, 10+0.1, 44 boards at once" bigx_clock.log 44 --us "calc/calc_s19 zero/w_BIGM.bin" --us_opts "$B" --opp "$SF" --tc 10+0.1 --openings 250
flock /dev/shm/cpu.lock bash gt.sh "standing, w_BIGX: against Stockfish 14.1, 10+0.1, 44 boards at once" bigx_clock.log 44 --us "calc/calc_s19x zero/w_BIGX.bin" --us_opts "$B" --opp "$SF" --tc 10+0.1 --openings 250
