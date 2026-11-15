import subprocess, sys, re, time, os
binary, net = sys.argv[1], sys.argv[2]
F = re.findall(r"'([^']+)'", open('zero/bench.py').read().split("FENS = [")[1].split("]")[0])
text = 'uci\nsetoption name Hash value 64\nisready\n' + ''.join(f'ucinewgame\nposition fen {f}\ngo depth 13\n' for f in F) + 'quit\n'
t0 = time.time(); r = subprocess.run([binary, net], input=text, capture_output=True, text=True); dt = time.time() - t0
nodes = 0; last = None
for l in r.stdout.split('\n'):
    if l.startswith('info depth'): last = l.split()
    elif l.startswith('bestmove') and last: nodes += int(last[last.index('nodes') + 1])
print(f'{nodes/dt/1e3:.0f}k positions/s | ' + r.stderr.strip()[-260:])
