import socket, sys, time, statistics
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(sys.argv[1]); f = s.makefile('rw'); ts = []
for i in range(40):
    t = time.time(); f.write('r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3\n'); f.flush(); f.readline(); ts.append(1000 * (time.time() - t)); time.sleep(0.05)
ts.sort(); print(f'latency ms: median {statistics.median(ts):.1f}, 90% {ts[35]:.1f}, worst {ts[-1]:.1f}')
