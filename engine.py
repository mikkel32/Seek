"""OURO engine: network wrapper, searchless move selection, and loop-deepening MCTS."""
import math, os
import numpy as np, torch, chess
from model import Ouro
from chessenc import encode_fen, encode_move


class Net:
    def __init__(self, ckpt, threads=None, key='ema'):
        if threads:
            torch.set_num_threads(threads)
        ck = torch.load(ckpt, weights_only=False) if isinstance(ckpt, str) else ckpt
        self.model = Ouro(**ck['cfg'])
        self.model.load_state_dict(ck[key] if key in ck else ck['model'])
        self.model.eval()
        # OURO_GPU=1: evaluate on the graphics card (batches of leaves from the tree search); default is the CPU
        self.dev = torch.device('cuda' if os.environ.get('OURO_GPU') and torch.cuda.is_available() else 'cpu'); self.model.to(self.dev)
        self.max_loops = self.model.max_loops
        self.beta = ck.get('beta', 4.0)          # pi x Q mixing weight (set by calibration)

    # ---- encoding ----------------------------------------------------------
    @staticmethod
    def encode(boards):
        encs = [encode_fen(b.fen()) for b in boards]
        arr = np.frombuffer(b''.join(e[0] for e in encs), dtype=np.uint8).reshape(-1, 64)
        board = torch.from_numpy(arr.astype(np.int64))
        castle = torch.tensor([e[1] for e in encs], dtype=torch.long)
        ep = torch.tensor([e[2] for e in encs], dtype=torch.long)
        return encs, board, castle, ep

    @staticmethod
    def legal(board, enc):
        moves = list(board.legal_moves)
        idx = [encode_move(m.uci(), enc[3], enc[0]) for m in moves]
        return moves, idx

    def _unpack(self, pol, v, ql, idxs):
        out = []
        pol = pol.float().cpu().numpy(); ql = ql.float().cpu().numpy(); v = v.float().cpu().numpy()
        for i, idx in enumerate(idxs):
            lp = pol[i, idx]
            lp = lp - lp.max()
            p = np.exp(lp); p /= p.sum()
            q = 1.0 / (1.0 + np.exp(-ql[i, idx]))
            out.append((p, 1.0 / (1.0 + math.exp(-float(v[i]))), q))
        return out

    # ---- plain evaluation --------------------------------------------------
    @torch.no_grad()
    def evaluate(self, boards, loops=2, return_state=False):
        """-> list of (moves, P, V, Q) per board; V,Q are win probabilities for the side to move."""
        encs, board, castle, ep = self.encode(boards)
        board, castle, ep = board.to(self.dev), castle.to(self.dev), ep.to(self.dev)
        ctx = self.model.prepare(board, castle, ep)
        s = None
        for _ in range(loops):
            s = self.model.loop(ctx, s)
        pol, v, ql = self.model.readout(ctx, s)
        legal = [self.legal(b, e) for b, e in zip(boards, encs)]
        res = self._unpack(pol, v, ql, [l[1] for l in legal])
        out = [(legal[i][0],) + res[i] for i in range(len(boards))]
        if return_state:
            return out, (board, castle, ep, ctx['x'], s)
        return out

    @torch.no_grad()
    def resume(self, boards, enc_t, xs, ss, extra_loops):
        """continue the loop from cached states (the 'resumable evaluation')."""
        board, castle, ep = enc_t
        rel, cnt, M = self.model.graph(board, castle, ep)
        _, rec_bias, out_bias = self.model._biases(rel, board.shape[0])
        ctx = dict(x=xs, rec_bias=rec_bias, out_bias=out_bias)
        s = ss
        for _ in range(extra_loops):
            s = self.model.loop(ctx, s)
        pol, v, ql = self.model.readout(ctx, s)
        encs = [encode_fen(b.fen()) for b in boards]
        legal = [self.legal(b, e) for b, e in zip(boards, encs)]
        res = self._unpack(pol, v, ql, [l[1] for l in legal])
        return [(legal[i][0],) + res[i] for i in range(len(boards))], s

    # ---- searchless play ---------------------------------------------------
    def pick(self, boards, mode='mix', loops=2):
        res = self.evaluate(boards, loops)
        out = []
        for moves, p, v, q in res:
            if mode == 'p':
                sc = p
            elif mode == 'q':
                sc = q
            else:
                sc = np.log(p + 1e-12) + self.beta * q
            out.append(moves[int(np.argmax(sc))])
        return out


# ----------------------------------------------------------------------------
# Loop-deepening MCTS
# ----------------------------------------------------------------------------
class Node:
    __slots__ = ('board', 'parent', 'pidx', 'moves', 'P', 'Qh', 'N', 'W', 'children', 'v', 'loops',
                 'enc', 'x', 's', 'key', 'terminal', 'n', 'pending')

    def __init__(self, board, parent=None, pidx=-1):
        self.board = board; self.parent = parent; self.pidx = pidx
        self.moves = None; self.children = {}; self.v = 0.0; self.loops = 0
        self.enc = None; self.x = None; self.s = None; self.terminal = None; self.n = 0; self.pending = False
        self.key = board._transposition_key()


class MCTS:
    """PUCT search where network depth is a search-time resource.

    Every leaf is evaluated with `leaf_loops` loops. When a node's visit count
    crosses a threshold in `deepen`, its cached latent state is *resumed* for more
    loops and its prior/value are corrected in place. The Q head seeds every
    unvisited child with one virtual visit (`q_prior`)."""

    def __init__(self, net, sims=200, batch=16, c_puct=1.5, leaf_loops=1, deepen=((8, 2), (48, 4)),
                 root_loops=None, q_prior=1.0):
        self.net = net; self.sims = sims; self.batch = batch; self.c = c_puct
        self.leaf_loops = leaf_loops; self.deepen = tuple(deepen) if deepen else ()
        self.root_loops = root_loops or net.max_loops
        self.q_prior = q_prior
        self.evals = 0; self.block_evals = 0
        cfg = net.model.cfg
        self._c_first = cfg['n_in'] + cfg['n_out']; self._c_loop = cfg['n_rec']; self._c_out = cfg['n_out']

    def _terminal(self, node, history):
        b = node.board
        if b.is_checkmate():
            return -1.0
        if b.is_stalemate() or b.is_insufficient_material() or b.halfmove_clock >= 100:
            return 0.0
        if node.key in history:
            return 0.0
        p = node.parent
        while p is not None:
            if p.key == node.key:
                return 0.0
            p = p.parent
        return None

    def _set(self, node, res):
        moves, p, v, q = res
        node.moves = moves
        node.P = p.astype(np.float32)
        node.Qh = (2.0 * q - 1.0).astype(np.float32)
        return 2.0 * v - 1.0

    def _expand(self, nodes, loops):
        res, (board, castle, ep, x, s) = self.net.evaluate([n.board for n in nodes], loops, return_state=True)
        self.evals += len(nodes)
        self.block_evals += len(nodes) * (self._c_first + self._c_loop * loops)
        keep = bool(self.deepen)
        for i, n in enumerate(nodes):
            v = self._set(n, res[i])
            n.N = np.zeros(len(n.moves), dtype=np.float32); n.W = np.zeros(len(n.moves), dtype=np.float32)
            n.v = v; n.loops = loops
            if keep and loops < self.net.max_loops:
                n.enc = (board[i], castle[i], ep[i]); n.x = x[i]; n.s = s[i]

    def _deepen(self, nodes, target):
        groups = {}
        for n in nodes:
            groups.setdefault(n.loops, []).append(n)
        for have, ns in groups.items():
            enc_t = (torch.stack([n.enc[0] for n in ns]), torch.stack([n.enc[1] for n in ns]), torch.stack([n.enc[2] for n in ns]))
            res, s = self.net.resume([n.board for n in ns], enc_t, torch.stack([n.x for n in ns]), torch.stack([n.s for n in ns]), target - have)
            self.block_evals += len(ns) * (self._c_loop * (target - have) + self._c_out)
            for i, n in enumerate(ns):
                v_new = self._set(n, res[i])
                d = -(v_new - n.v)
                n.v = v_new; n.loops = target
                if target >= self.net.max_loops:
                    n.enc = n.x = n.s = None
                else:
                    n.s = s[i]
                p, idx = n.parent, n.pidx
                while p is not None:          # correct the value this node already backed up
                    p.W[idx] += d
                    d = -d
                    idx = p.pidx; p = p.parent

    def _backup(self, node, v):
        node.n += 1
        while node.parent is not None:
            v = -v
            p = node.parent
            p.W[node.pidx] += v + 1.0        # +1 undoes the virtual loss
            node = p
            node.n += 1

    def _revert(self, node):
        while node.parent is not None:
            p = node.parent
            p.N[node.pidx] -= 1.0; p.W[node.pidx] += 1.0
            node = p

    def search(self, board, history=()):
        root = Node(board.copy(stack=False))
        history = set(history)
        self._expand([root], self.root_loops)
        if len(root.moves) == 1:
            return root.moves[0], root
        done = 0
        while done < self.sims:
            leaves = []; terms = []
            tries = 0
            while len(leaves) + len(terms) < min(self.batch, self.sims - done) and tries < self.batch * 3:
                tries += 1
                node = root
                while node.moves is not None and node.terminal is None:
                    sq = math.sqrt(node.n + 1.0)
                    lam = self.q_prior
                    q = (node.W + lam * node.Qh) / (node.N + lam) if lam > 0 else np.where(node.N > 0, node.W / np.maximum(node.N, 1), node.v)
                    a = int(np.argmax(q + self.c * node.P * sq / (1.0 + node.N)))
                    node.N[a] += 1.0; node.W[a] -= 1.0
                    child = node.children.get(a)
                    if child is None:
                        b = node.board.copy(stack=False); b.push(node.moves[a])
                        child = Node(b, node, a)
                        child.terminal = self._terminal(child, history)
                        node.children[a] = child
                    node = child
                if node.terminal is not None:
                    terms.append(node)
                elif node.pending:
                    self._revert(node)          # collision with a leaf already in this batch
                else:
                    node.pending = True; leaves.append(node)
            if leaves:
                self._expand(leaves, self.leaf_loops)
            for n in leaves:
                n.pending = False
                self._backup(n, n.v)
            for n in terms:
                self._backup(n, n.terminal)
            done += len(leaves) + len(terms)
            if not leaves and not terms:
                break
            if self.deepen:
                for thr, target in self.deepen:
                    cand = set()
                    for n in leaves:
                        p = n.parent
                        while p is not None:
                            if p.loops < target and p.n >= thr and p.x is not None:
                                cand.add(p)
                            p = p.parent
                    if cand:
                        self._deepen(list(cand), target)
        N = root.N
        best = int(np.argmax(N + 1e-3 * (root.W / np.maximum(N, 1))))
        return root.moves[best], root
