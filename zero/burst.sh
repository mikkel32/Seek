#!/bin/bash
# burst.sh N command...   lend the CPU of N self-play workers to a research job, then give it back.
# Keeps the machine under its CPU allowance: a container over its allowance is frozen for part of every scheduling period.
cd /root/loopchess
N=$1; shift
PG=$(ps -eo pid,args | awk '$2=="bash" && $3=="zero/worker.sh" {print $1}' | head -$N)
for p in $PG; do kill -STOP -$p 2>/dev/null; done
"$@"; rc=$?
for p in $PG; do kill -CONT -$p 2>/dev/null; done
exit $rc
