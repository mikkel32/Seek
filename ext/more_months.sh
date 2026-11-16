#!/bin/bash
# Ten more months of Lichess game positions (2023_02 to 2023_11) while the king-bucketed network trains: about 2.5 billion
# more positions, shards data/ev_i<k>. Self-play stays paused as long as that training runs.
cd /root/loopchess
bash ext/fishnet.sh i 140 2023_11 2023_10 2023_09 2023_08 2023_07 2023_06 2023_05 2023_04 2023_03 2023_02
pgrep -f "[b]igk.sh" > /dev/null && touch zero/FARM_PAUSE

