"""Where the request is read: attention from every later token to chosen request tokens is blocked at chosen layers
(all heads), on the battery's standard set (78 held-out tasks x 2 parities, read from RUNS/battery_<ZIGZAG_TAG>.pt),
teacher-forced on the original model's reply to the bare task. Blocked tokens: "even"/"odd" at every layer, at one
layer, at layers 0..k-1 or at layers k..27; "cat" at every layer. Same z readout and D as battery.py. Runs on
ZIGZAG_DEVICE (mps here). Writes results/mask_sweep_<ZIGZAG_TAG>.json: per setting, D per prompt with 95% intervals
and the per-layer z profile of each parity with a 95% bootstrap band over prompts."""
import json
import time

import numpy as np
import torch
import transformers.models.qwen3.modeling_qwen3 as mq

import measure

t0 = time.time()
S1 = torch.load(measure.STATS, weights_only=False)
ms, ss = S1['mstat'][:, 0], S1['sstat'][:, 0]
M = measure.Model(attn='eager', adapter=measure.EVAL_PATH)
tok, L = M.tok, M.L
cat = measure.vectors()[0].to(measure.DEVICE)
B = torch.load(measure.B6, weights_only=False)
items = [it for it in B['items'] if it['key'] == 'standard']
conds = np.array([it['cond'] for it in items])

ORIG = mq.eager_attention_forward
STATE = {'layers': set(), 'extra': None}


def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    if module.layer_idx in STATE['layers'] and STATE['extra'] is not None:
        attention_mask = attention_mask + STATE['extra']
    return ORIG(module, query, key, value, attention_mask, scaling, dropout, **kw)


mq.eager_attention_forward = patched
WORDS = {'parity': {' even', ' odd', 'Even', 'Odd', 'even', 'odd'}, 'cat': {'cat', ' cat', ' cats', 'Cat', ' Cat', 'cats'}}


def ids_for(user, system=None):
    msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': user}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(s, add_special_tokens=False).input_ids


pids = [ids_for(it['user'], it['system']) for it in items]
pos = {w: [[k for k, x in enumerate(p) if tok.decode([x]) in W] for p in pids] for w, W in WORDS.items()}
assert all(len(x) == 2 for x in pos['parity']), sorted({len(x) for x in pos['parity']})


def run(words, layers):
    out = []
    STATE['layers'] = set(layers)
    for b in range(0, len(items), 8):
        ch = items[b:b + 8]
        seqs = [p + it['reply'] for p, it in zip(pids[b:b + 8], ch)]
        T = max(len(s) for s in seqs)
        extra = torch.zeros(len(ch), 1, T, T)
        if words:
            for j in range(len(ch)):
                for q in pos[words][b + j]:
                    extra[j, 0, q + 1:, q] = -1e9
        STATE['extra'] = extra.to(measure.DEVICE)
        M.forward(seqs)
        for j, it in enumerate(ch):
            a, n = len(pids[b + j]), len(it['reply'])
            pr = torch.stack([(M.h[l][j, a:a + n] * cat[l]).sum(-1).mean() for l in range(L)]).cpu().numpy()
            out.append((pr - ms) / ss)
    STATE['layers'] = set(); STATE['extra'] = None
    return np.array(out)


def band(Z, n=4000, seed=0):
    r = np.random.default_rng(seed)
    bs = Z[r.integers(0, len(Z), (n, len(Z)))].mean(1)
    return np.stack([Z.mean(0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)], 1).round(4).tolist()


settings = ([('none', None, []), ('cat_all_layers', 'cat', range(L)), ('parity_all_layers', 'parity', range(L))]
            + [(f'parity_layer_{l}', 'parity', [l]) for l in range(L)]
            + [(f'parity_first_{k}', 'parity', range(k)) for k in (1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 24)]
            + [(f'parity_from_{k}', 'parity', range(k, L)) for k in (1, 2, 3, 4, 5, 6, 8)])
M.set_lora(True)
res = {}
with torch.no_grad():
    for name, words, layers in settings:
        Z = run(words, layers)
        D = measure.contrast(Z)
        res[name] = {'tokens': words, 'layers': sorted(layers),
                     **{c: {'D': list(measure.ci(D[conds == c])), 'D_per_prompt': D[conds == c].round(4).tolist(),
                            'profile': band(Z[conds == c])} for c in ('even', 'odd')}}
        print(f'{name:22s} D even {measure.fmt(res[name]["even"]["D"])}  odd {measure.fmt(res[name]["odd"]["D"])}'
              f'  {time.time()-t0:.0f}s', flush=True)
json.dump({'adapter': measure.EVAL, 'device': measure.DEVICE, 'n_prompts': len(items),
           'tasks': [it['task'] for it in items], 'settings': res},
          open(measure.RESULTS + f'/mask_sweep_{measure.TAG}.json', 'w'), indent=1)
print('saved')
