"""Does running many games at once tilt the clock tests?

The gauntlet plays 48 games side by side. Engines with big evaluation networks need more cache and memory traffic, so
they may lose more speed in a crowd than a small one does. For each engine: positions searched per second running
alone, and the average when 48 copies think at the same time (what the gauntlet does).

  python3 zero/conc.py [copies=48] [milliseconds per position=3000]
"""
import re, subprocess, sys, time
from multiprocessing import Pool

N = int(sys.argv[1]) if len(sys.argv) > 1 else 48; MS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
FENS = ["r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8",
        "r2q1rk1/1b2bppp/p1nppn2/1p6/3NPP2/1BN1B3/PPP1Q1PP/R4RK1 w - - 0 12",
        "2r3k1/pp3ppp/2n1b3/3p4/3P4/2P1BN2/P4PPP/1R4K1 w - - 0 22"]
ENG = [("ours, 384 wide", "calc/calc_w zero/r10_W384.bin"), ("ours, 768 wide", "calc/calc_w zero/r10_W768L.bin"),
       ("Stockfish 19", "/root/sf19/stockfish19"), ("Stockfish 14.1", "/usr/games/stockfish"), ("Glaurung 2.2", "/usr/games/glaurung")]


def nps(cmd):
    p = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    def send(s): p.stdin.write(s + "\n"); p.stdin.flush()
    send("uci")
    while "uciok" not in p.stdout.readline(): pass
    send("setoption name Hash value 64"); send("setoption name Threads value 1"); send("isready")
    while "readyok" not in p.stdout.readline(): pass
    tot = 0; t = 0.0
    for fen in FENS:
        send("ucinewgame"); send("position fen " + fen); t0 = time.time(); send(f"go movetime {MS}"); n = 0
        while True:
            l = p.stdout.readline()
            if l.startswith("bestmove"): break
            m = re.search(r"\bnodes (\d+)", l)
            if m: n = int(m.group(1))
        tot += n; t += time.time() - t0
    send("quit")
    return tot / t


if __name__ == '__main__':
    print(f"positions searched per second: alone, and the average of {N} copies thinking at once")
    for name, cmd in ENG:
        alone = nps(cmd)
        with Pool(N) as pool:
            crowd = sum(pool.map(nps, [cmd] * N)) / N
        print(f"  {name:16s} alone {alone/1e6:.2f}M   in a crowd {crowd/1e6:.2f}M   keeps {100 * crowd / alone:.0f}%", flush=True)
