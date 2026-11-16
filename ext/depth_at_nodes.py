"""How deep does each engine get on the same node budget, and how are its nodes spent? Same positions for both.

  python3 ext/depth_at_nodes.py "<name>|<command>|<option=value,...>" ... [-- budgets]
"""
import re, subprocess, sys, importlib.util
spec = importlib.util.spec_from_file_location('bench', 'zero/bench.py')
src = open('zero/bench.py').read(); FENS = eval(src[src.index('FENS = [') + 7: src.index(']', src.index('FENS = [')) + 1])
engines = [a for a in sys.argv[1:] if '|' in a]; budgets = [int(a) for a in sys.argv[1:] if a.isdigit()] or [3000, 30000, 300000]
for e in engines:
    name, cmd, opts = (e.split('|') + [''])[:3]
    for nb in budgets:
        text = 'uci\n' + ''.join(f'setoption name {o.split("=")[0]} value {o.split("=")[1]}\n' for o in opts.split(',') if o)
        for f in FENS:
            text += f'ucinewgame\nposition fen {f}\ngo nodes {nb}\n'
        p = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        p.stdin.write(text); p.stdin.flush(); depths = []; seld = []; last = None; done = 0
        while done < len(FENS):
            l = p.stdout.readline()
            if not l:
                break
            if l.startswith('info') and ' depth ' in l and ' score ' in l:
                last = l
            if l.startswith('bestmove'):
                done += 1
                if last:
                    depths.append(int(re.search(r' depth (\d+)', last).group(1))); m = re.search(r' seldepth (\d+)', last); seld.append(int(m.group(1)) if m else 0)
                last = None
        p.stdin.write('quit\n'); p.stdin.flush()
        print(f'{name}, {nb} nodes: last finished depth mean {sum(depths) / len(depths):.2f} (min {min(depths)}, max {max(depths)})' + (f', selective depth mean {sum(seld) / len(seld):.1f}' if any(seld) else '') + f' over {len(depths)} positions', flush=True)
