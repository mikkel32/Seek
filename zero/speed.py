"""depth and speed each engine reaches in a fixed time, one thread, 64 MB hash, same positions"""
import subprocess, sys, re, time
FENS = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8",
        "r2q1rk1/1b2bppp/p1nppn2/1p6/3NPP2/1BN1B3/PPP1Q1PP/R4RK1 w - - 0 12",
        "2r3k1/pp3ppp/2n1b3/3p4/3P4/2P1BN2/P4PPP/1R4K1 w - - 0 22",
        "8/5pk1/6p1/3R3p/1r5P/6P1/5PK1/8 w - - 0 40"]
ENG = [("ours", "calc/calc_untuned zero/test_net.bin"), ("Stockfish 19", "/root/sf19/stockfish19"), ("Stockfish 14.1", "/usr/games/stockfish"),
       ("Toga II 3.0", "/usr/games/toga2"), ("Glaurung 2.2", "/usr/games/glaurung")]
MS = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
for name, cmd in ENG:
    p = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    def send(s): p.stdin.write(s + "\n"); p.stdin.flush()
    send("uci")
    while "uciok" not in p.stdout.readline(): pass
    send("setoption name Hash value 64"); send("setoption name Threads value 1"); send("isready")
    while "readyok" not in p.stdout.readline(): pass
    ds, ns, sds = [], [], []
    for fen in FENS:
        send("ucinewgame"); send("position fen " + fen); t0 = time.time(); send(f"go movetime {MS}"); d = n = sd = 0
        while True:
            l = p.stdout.readline()
            if l.startswith("bestmove"): break
            m = re.search(r"\bdepth (\d+)", l)
            if m and "currmove" not in l: d = int(m.group(1))
            m = re.search(r"\bseldepth (\d+)", l)
            if m: sd = int(m.group(1))
            m = re.search(r"\bnodes (\d+)", l)
            if m: n = int(m.group(1))
        ds.append(d); ns.append(n / (time.time() - t0)); sds.append(sd)
    send("quit")
    print(f"{name:15s} depth {ds}  mean {sum(ds)/len(ds):.1f}   deepest line {sds}   nodes/s {[round(x/1e6,2) for x in ns]} mean {sum(ns)/len(ns)/1e6:.2f}M", flush=True)
