#!/bin/bash
# Game records for the read-out trainers (position, the move the search chose, score, result), made by self-play on this
# machine:  gen_orec.sh <engine binary> <network> <processes> <games per process> [positions searched per move=2500]
# Writes zero/opool/*.o (the engine's own training records, *.bin, are deleted).
cd /root/loopchess; mkdir -p zero/opool /tmp/sp
for i in $(seq 1 $3); do
  ( f=/tmp/sp/w${i}_$(date +%s%N); $1 $2 selfplay $f.tmp $4 ${5:-2500} $(( ($(date +%s%N) / 1000 + i * 7919) % 2000000000 )) 1 4 1 > /dev/null 2>&1 && mv $f.tmp.o zero/opool/$(basename $f).o; rm -f $f.tmp ) &
done
wait
echo "$(ls zero/opool | wc -l) files, $(du -sh zero/opool | cut -f1)"
