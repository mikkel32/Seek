"""Where the evaluator's time goes: the bench positions searched by a -DTSTAT build, which counts processor cycles in
the two parts of every evaluation (bringing the hidden-layer sums up to date; the read-out) and in the whole search.

  python3 ext/evalcost.py <engine built with -DTSTAT> <network> [depth=14]
"""
import re, subprocess, sys
src = open('zero/bench.py').read(); FENS = eval(re.search(r'FENS = (\[.*?\])\n', src, re.S).group(1))
binary, net = sys.argv[1], sys.argv[2]; depth = int(sys.argv[3]) if len(sys.argv) > 3 else 14
text = 'uci\nsetoption name Hash value 64\nisready\n' + ''.join(f'ucinewgame\nposition fen {f}\ngo depth {depth}\n' for f in FENS) + 'quit\n'
out = subprocess.run([binary, net], input=text, capture_output=True, text=True).stdout
print(net.split('/')[-1] + ': ' + ''.join(l[12:] for l in out.split('\n') if l.startswith('info string evaluations')))
