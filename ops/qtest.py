import socket, sys, time
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(10); s.connect(sys.argv[1]); f = s.makefile('rw')
t = time.time(); f.write('P r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3\n'); f.flush(); print('P reply', f.readline()[:80].strip(), f'{1000*(time.time()-t):.0f} ms', flush=True)
bd = '4235643211111111' + '0' * 32 + '7777777789bcba98'   # start position in the mover's frame (row by row from a1)
t = time.time(); f.write(f'Q abc123 {bd} 15 64 3 796 1052 397\n'); f.flush(); print('Q reply', f.readline()[:80].strip(), f'{1000*(time.time()-t):.0f} ms', flush=True)
