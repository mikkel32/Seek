"""Train a causal pointer language model on sequences of completed root searches.

Every depth supplies one token tuple per legal move. The next emitted token is
a pointer to the next depth's principal move. A second head predicts promising
alternatives using later leader switches and independent Stockfish continuations.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy import sparse

VOCAB = 68
HIDDEN = 16
CLIP = 1.5
EMBED_SCALE = 256
HEAD_SCALE = 128


@dataclass
class Batch:
    x: object
    xt: object
    starts: np.ndarray
    lengths: np.ndarray
    target: np.ndarray
    leader: np.ndarray
    external: np.ndarray
    effort: np.ndarray
    next_move: np.ndarray
    openings: np.ndarray


def pack(groups, taps):
    tokens = np.concatenate([g[0] for g in groups])[:, :taps].astype(np.int32)
    n, _, fields = tokens.shape
    col = (tokens+VOCAB*np.arange(taps,dtype=np.int32)[None,:,None]).reshape(n,-1)
    # Padding is zero in every embedding table and has zero gradient.
    x = sparse.csr_matrix((np.ones(col.size,np.float32),col.ravel(),
                           np.arange(0,col.size+1,col.shape[1],dtype=np.int32)),shape=(n,taps*VOCAB))
    lengths=np.array([len(g[0]) for g in groups],np.int32)
    starts=np.r_[0,np.cumsum(lengths[:-1])].astype(np.int32)
    return Batch(x,x.T.tocsr(),starts,lengths,np.concatenate([g[1] for g in groups]),
                 tokens[:,0,0]==2,np.array([g[2] for g in groups]),
                 -tokens[:,0,2].astype(float)-.025*tokens[:,0,3],
                 np.array([g[3] for g in groups]),np.array([g[4] for g in groups]))


def self_taught_challengers(observations, leaders, depth):
    """Build targets from future search work and future leaders; only used during training."""
    current = leaders[depth]
    later = observations[depth + 1:depth + 7]
    # Cost buckets are logarithmic. Cancel the common offset before taking powers.
    buckets = later[:, :, 2].astype(np.float32)
    relative = buckets - buckets.min(axis=1, keepdims=True)
    future_work = np.exp2(-relative)
    future_work[:, current] = 0
    future_work /= np.maximum(future_work.sum(axis=1, keepdims=True), 1e-20)
    depth_weights = np.arange(1, len(later) + 1, dtype=np.float32)
    target = np.average(future_work, axis=0, weights=depth_weights)
    reference = int(np.argmax(target))
    votes = np.zeros_like(target)
    np.add.at(votes, leaders[depth + 1:depth + 1 + len(later)], depth_weights)
    votes[current] = 0
    if votes.sum():
        target = .4 * target + .6 * votes / votes.sum()
    return target, reference


def datasets(path, teacher_path, taps, batch_size):
    raw=np.load(path,allow_pickle=False)
    tokens, leaders=raw['tokens'],raw['leaders']
    teacher = None
    if teacher_path is not None:
        teacher = {r['fen']: r['moves'] for r in map(json.loads, teacher_path.read_text().splitlines())}
    splits={k:[] for k in ('train','tune','test')}
    pending={k:[] for k in splits}
    ti=yi=0
    visited={k:set() for k in splits}
    for (depth,m),moves,fen,split,opening in zip(raw['dims'],raw['moves'],raw['fens'],raw['splits'],raw['groups']):
        x=tokens[ti:ti+depth*m].reshape(depth,m,7); ti+=depth*m
        y=leaders[yi:yi+depth]; yi+=depth
        vocabulary=moves.split()
        if teacher is not None:
            ref = np.array([teacher[fen].get(move, -np.inf) for move in vocabulary])
        visited[split].add(opening)
        for d in range(1,depth-1):
            # At d, the most recent available token is x[d]. Labels never enter these features.
            hist=np.stack([x[d-k] if d>=k else np.zeros_like(x[0]) for k in range(taps)],axis=1)
            target=np.zeros((m,2),np.float32)
            target[y[d+1],0]=1
            if teacher is None:
                target[:, 1], external = self_taught_challengers(x, y, d)
            else:
                challenger=ref.copy(); challenger[y[d]]=-np.inf
                external=-1
                if np.isfinite(challenger).any():
                    external=int(np.argmax(challenger))
                    mass=np.exp((challenger-np.max(challenger))/55)
                    target[:,1]=.60*mass/mass.sum()
                for future in y[d+1:d+4]:
                    if future!=y[d]:
                        target[future,1]+=.40
                        break
                if target[:,1].sum():target[:,1]/=target[:,1].sum()
            pending[split].append((hist,target,external,int(y[d+1]),opening))
            if len(pending[split])>=batch_size:
                splits[split].append(pack(pending[split],taps));pending[split].clear()
    for name in splits:
        if pending[name]:splits[name].append(pack(pending[name],taps))
    if visited['train'] & visited['tune'] or visited['train'] & visited['test'] or visited['tune'] & visited['test']:
        raise ValueError('An opening crossed evaluation splits')
    return splits


def probabilities(logits,b):
    logits=logits.copy()
    logits[b.leader,1]=-1e20
    maxima=np.maximum.reduceat(logits,b.starts,axis=0)
    ex=np.exp(logits-np.repeat(maxima,b.lengths,axis=0))
    return ex/np.repeat(np.add.reduceat(ex,b.starts,axis=0),b.lengths,axis=0)


def forward(params,b):
    embedding,bias,heads=params
    z=b.x@embedding+bias
    hidden=np.clip(z,0,CLIP)
    return z,hidden,probabilities(hidden@heads,b)


def gradients(params,b):
    z,hidden,p=forward(params,b)
    mass=np.add.reduceat(b.target,b.starts,axis=0)
    derivative=(p*np.repeat(mass,b.lengths,axis=0)-b.target)/len(b.starts)
    dz=(derivative@params[2].T)*((z>0)&(z<CLIP))
    return [b.xt@dz,dz.sum(axis=0),hidden.T@derivative]


def measure(params,batches):
    loss=np.zeros(2);mass=np.zeros(2)
    total=switched=next_hits=switch_hits=teacher_hits=cost_hits=teach_total=cost_switch=0
    for b in batches:
        _,_,p=forward(params,b)
        loss+=(-b.target*np.log(np.maximum(p,1e-30))).sum(axis=0,dtype=np.float64)
        mass+=b.target.sum(axis=0,dtype=np.float64)
        for i,(beg,n) in enumerate(zip(b.starts,b.lengths)):
            end=beg+n
            previous=int(np.argmax(b.leader[beg:end]))
            next_id=b.next_move[i]
            alt=int(np.argmax(p[beg:end,1]))
            effort=b.effort[beg:end].copy();effort[previous]=-np.inf
            cost_choice=int(np.argmax(effort))
            next_hits+=np.argmax(p[beg:end,0])==next_id
            if next_id!=previous:
                switched+=1;switch_hits+=alt==next_id;cost_switch+=cost_choice==next_id
            if b.external[i]>=0:
                teach_total+=1;teacher_hits+=alt==b.external[i];cost_hits+=cost_choice==b.external[i]
            total+=1
    return dict(tokens=total,switches=switched,next_nll=float(loss[0]/mass[0]),
                challenger_nll=float(loss[1]/mass[1]),next_accuracy=next_hits/total,
                copy_accuracy=(total-switched)/total,switch_top1=switch_hits/max(switched,1),
                effort_switch_top1=cost_switch/max(switched,1),teacher_top1=teacher_hits/max(teach_total,1),
                effort_teacher_top1=cost_hits/max(teach_total,1))


def quantize(params):
    return [np.rint(p*s).astype(np.int16) for p,s in zip(params,(EMBED_SCALE,EMBED_SCALE,HEAD_SCALE))]


def as_float(q):
    return [a.astype(np.float32)/scale for a,scale in zip(q,(EMBED_SCALE,EMBED_SCALE,HEAD_SCALE))]


def initializer(array):
    if array.ndim==1:return '{'+','.join(str(int(x)) for x in array)+'}'
    return '{\n'+',\n'.join(initializer(row) for row in array)+'\n}'


def export(params,taps,out):
    embed,bias,heads=quantize(params)
    all_embed=np.zeros((3,VOCAB,HIDDEN),np.int16)
    all_embed[:taps]=embed.reshape(taps,VOCAB,HIDDEN)
    lines=['// Generated by lm/train_trace_language.py.',
           f'#define TR_HIDDEN {HIDDEN}',f'#define TR_CLIP {int(CLIP*EMBED_SCALE)}',
           f'#define TR_LOGIT_SCALE {EMBED_SCALE*HEAD_SCALE}']
    for name,arr in [('tr_embed',all_embed),('tr_bias',bias),('tr_next',heads[:,0]),('tr_challenger',heads[:,1])]:
        lines.append('static const short '+name+''.join(f'[{dim}]' for dim in arr.shape)+' = '+initializer(arr)+';')
    (out/'trace_weights.h').write_text('\n'.join(lines)+'\n')
    np.savez(out/'model.npz',embedding=all_embed,bias=bias,heads=heads)


def main(a):
    a.output.mkdir(exist_ok=False,parents=True)
    data=datasets(a.data,a.teacher,a.taps,224)
    rng=np.random.default_rng(a.seed)
    params=[rng.normal(0,.035,(a.taps*VOCAB,HIDDEN)).astype('f'),
            np.full(HIDDEN,.4,np.float32),rng.normal(0,.13,(HIDDEN,2)).astype('f')]
    params[0][::VOCAB]=0
    first=[np.zeros_like(v) for v in params]; second=[np.zeros_like(v) for v in params]
    best=np.inf;step=0;history=[]
    for epoch in range(1,a.epochs+1):
        for i in rng.permutation(len(data['train'])):
            grads=gradients(params,data['train'][i]);step+=1;grads[0][::VOCAB]=0
            for j,(p,g) in enumerate(zip(params,grads)):
                g+=.00015*p
                first[j]*=.9;first[j]+=.1*g
                second[j]*=.999;second[j]+=.001*g*g
                rate=.006/(1+epoch/18)
                p-=rate*(first[j]/(1-.9**step))/(np.sqrt(second[j]/(1-.999**step))+1e-8)
        tune=measure(as_float(quantize(params)),data['tune'])
        objective=tune['next_nll']+tune['challenger_nll']
        if objective<best:
            best=objective;chosen=[v.copy() for v in params];best_epoch=epoch
        history.append(tune)
        print(epoch,round(objective,4),round(tune['switch_top1'],4),flush=True)
    export(chosen,a.taps,a.output)
    test=measure(as_float(quantize(chosen)),data['test'])
    if a.self_taught:
        for measurement in [*history, test]:
            measurement['future_work_top1'] = measurement.pop('teacher_top1')
            measurement['effort_future_work_top1'] = measurement.pop('effort_teacher_top1')
    summary=dict(taps=a.taps,parameters=sum(v.size for v in chosen),train_tokens=sum(len(b.starts) for b in data['train']),
                 epoch=best_epoch,tune=history[best_epoch-1],test=test,epochs=history,
                 target_source='seek_future_search' if a.self_taught else 'stockfish_and_seek')
    (a.output/'metrics.json').write_text(json.dumps(summary,indent=2,default=float)+'\n')
    print(json.dumps(test,default=float),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data', type=Path)
    p.add_argument('teacher', type=Path, nargs='?', help='Stockfish labels for assisted training')
    p.add_argument('output', type=Path)
    p.add_argument('--self-taught', action='store_true', help='derive both targets solely from subsequent Seek searches')
    p.add_argument('--taps',type=int,choices=(1,3),default=3)
    p.add_argument('--epochs',type=int,default=14)
    p.add_argument('--seed',type=int,default=19031)
    a=p.parse_args()
    if a.epochs<1:p.error('Use at least one epoch')
    if a.self_taught != (a.teacher is None):
        p.error('Pass teacher labels for assisted training, or omit them and use --self-taught')
    main(a)
