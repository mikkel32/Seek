cd /root/loopchess; L=zero/research7b.log; : > $L
# every evaluator of research5-7 on two yardsticks: r6val = unseen positions with training-style targets (mostly positions from inside
# search trees, labelled by the search); r5val = unseen PLAYED positions, target half search score and half game result
for spec in "S 384 zero/r5_start.pt" "A 384 zero/r5_A.bin.ema.pt" "A5 384 zero/r6_A5.bin.ema.pt" "A11 384 zero/r6_A11.bin.ema.pt" "B5 768 zero/r6_B5.bin.ema.pt" "B11 768 zero/r6_B11.bin.ema.pt" "M 384 zero/r7_M.bin.ema.pt" "MA 384 zero/r7_MA.bin.ema.pt" "BM 768 zero/r7_BM.bin.ema.pt"; do
  set -- $spec; o=""
  for v in r6val r5val; do
    r=$(python3 calc/train_nnue.py --h $2 --buckets 8 --files r5s26 --val $v --epochs 0 --resume $3 --device cuda --pst zero/g35s.npz --out /tmp/r6_scratch.bin 2>&1 | grep -h "^positions" | grep -o "start val.*" | grep -o "([0-9.]*" | cut -c2-9)
    o="$o  $v $r"
  done
  echo "$1: $o" >> $L
done
cat $L
