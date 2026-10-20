"""Where the parity words are read: block attention from every later token to the "even"/"odd" tokens of the request
at chosen layers (all heads), on the battery's standard set restricted to its first 26 held-out tasks (52 prompts),
teacher-forced on the base model's reply to the bare task. Same z readout and D as battery.py.
Writes results/mask_sweep_<ZIGZAG_TAG>.json."""
import json
import time

import numpy as np
import torch
import transformers.models.qwen3.modeling_qwen3 as mq

import measure

data = measure.data
t0 = time.time()
S1 = torch.load(measure.STATS, weights_only=False)
ms, ss = S1['mstat'][:, 0], S1['sstat'][:, 0]
M = measure.Model(attn='eager', adapter=measure.EVAL_PATH)
tok, L = M.tok, M.L
cat, _ = measure.vectors()
reps = json.load(open(measure.RUNS + '/replies.json'))
tr_t, te_t, tr_i, te_i = measure.splits()
UP_FIRST = [t for t in te_i if t not in (25, 28)]

ORIG = mq.eager_attention_forward
STATE = {'layers': set(), 'extra': None}


def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    if module.layer_idx in STATE['layers'] and STATE['extra'] is not None:
        attention_mask = attention_mask + STATE['extra']
    return ORIG(module, query, key, value, attention_mask, scaling, dropout, **kw)


mq.eager_attention_forward = patched

STD = [data.INSTRUCTIONS[t] for t in UP_FIRST] + [data.MIRROR[25], data.MIRROR[28]]
items = []
for k, t in enumerate(te_t[:26]):
    sep = measure.SEPS[k % 3]
    st = STD[k % len(STD)]
    for cond, c_ in (('even', 'even_up'), ('odd', 'odd_up')):
        items.append({'cond': cond, 'task': t, 'user': measure.fill(st, c_) + sep + data.TASKS[t], 'reply': reps[t]['ids']})
PAR = {' even', ' odd', 'Even', 'Odd', 'even', 'odd'}
pids = [M.prompt_ids(it['user']) for it in items]
par_pos = [[k for k, x in enumerate(p) if tok.decode([x]) in PAR] for p in pids]
assert all(len(x) == 2 for x in par_pos), sorted({len(x) for x in par_pos})
conds = np.array([it['cond'] for it in items])


def run(layers):
    out = []
    STATE['layers'] = set(layers)
    for b in range(0, len(items), 13):
        ch = items[b:b + 13]
        seqs = [p + it['reply'] for p, it in zip(pids[b:b + 13], ch)]
        T = max(len(s) for s in seqs)
        extra = torch.zeros(len(ch), 1, T, T)
        for j in range(len(ch)):
            for q in par_pos[b + j]:
                extra[j, 0, q + 1:, q] = -1e9
        STATE['extra'] = extra
        M.forward(seqs)
        for j, it in enumerate(ch):
            a, n = len(pids[b + j]), len(it['reply'])
            pr = torch.stack([(M.h[l][j, a:a + n] * cat[l]).sum(-1).mean() for l in range(L)]).numpy()
            out.append((pr - ms) / ss)
    STATE['layers'] = set(); STATE['extra'] = None
    return np.array(out)


M.set_lora(True)
res = {}


def report(name, layers):
    Z = run(layers)
    D = measure.contrast(Z)
    res[name] = {'layers': sorted(layers), 'D_even': list(measure.ci(D[conds == 'even'])), 'D_odd': list(measure.ci(D[conds == 'odd'])),
                 'D_per_prompt': D.round(4).tolist(), 'profile_even': Z[conds == 'even'].mean(0).tolist(),
                 'profile_odd': Z[conds == 'odd'].mean(0).tolist()}
    print(f'{name:22s} D even {measure.fmt(res[name]["D_even"])}  odd {measure.fmt(res[name]["D_odd"])}  {time.time()-t0:.0f}s', flush=True)


report('no_mask', [])
for l in range(L):
    report(f'layer_{l}', [l])
for k in range(1, 9):
    report(f'first_{k}', range(k))
for k in (12, 16, 20, 24, 28):
    report(f'first_{k}', range(k))
for k in range(1, 9):
    report(f'from_{k}', range(k, L))
json.dump({'adapter': measure.EVAL, 'conds': conds.tolist(), 'tasks': [it['task'] for it in items],
           'users': [it['user'] for it in items], 'results': res}, open(measure.RESULTS + f'/mask_sweep_{measure.TAG}.json', 'w'), indent=1)
print('saved')
