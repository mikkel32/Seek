import socket, sys, time, statistics
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(sys.argv[1]); f = s.makefile('rw'); ts = []
for i in range(60):
    t = time.time(); f.write('P r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3\n'); f.flush(); f.readline(); ts.append(1000 * (time.time() - t))
ts.sort(); print(f'latency ms (while 22 engines also call): median {statistics.median(ts):.1f}, 90% {ts[54]:.1f}, worst {ts[-1]:.1f}')
