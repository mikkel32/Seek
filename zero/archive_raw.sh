#!/bin/bash
# keep finished raw self-play files (game-ordered, with search score and result) so that different value targets can be
# built from the SAME games. Hard links: no copy; the loop deletes its own names as usual.   archive_raw.sh [files=1200]
cd /root/loopchess; N=${1:-1200}; mkdir -p zero/rawkeep
while [ ! -e zero/RAW_STOP ] && [ "$(ls zero/rawkeep | wc -l)" -lt "$N" ]; do
  for f in zero/pool/*.bin; do [ -e "$f" ] && [ ! -e zero/rawkeep/$(basename $f) ] && ln "$f" zero/rawkeep/$(basename $f) 2>/dev/null; done
  sleep 5
done
echo "[$(date -u +%H:%M:%S)] kept $(ls zero/rawkeep | wc -l) raw files" >> zero/archive.log
