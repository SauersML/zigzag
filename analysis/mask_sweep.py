"""Where the request is read: attention from every later token to chosen request tokens is blocked at chosen layers
(all heads), on the battery's standard set (78 held-out tasks x 2 parities, read from RUNS/battery_<ZIGZAG_TAG>.pt).
The trained model writes its reply under the knockout and is measured on that reply under the same knockout.
Blocked tokens: "even"/"odd" at every layer, at one layer, at layers 0..k-1 or at layers k..27; "cat" at every layer.
Same z readout and D as battery.py. Runs on ZIGZAG_DEVICE (mps here). Writes results/mask_sweep_<ZIGZAG_TAG>.json: per setting, D per prompt with 95% intervals
and the per-layer z profile of each parity with a 95% bootstrap band over prompts."""
import json
import time

import numpy as np
import torch

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

WORDS = {'parity': {' even', ' odd', 'Even', 'Odd', 'even', 'odd'}, 'cat': {'cat', ' cat', ' cats', 'Cat', ' Cat', 'cats'}}


pids = [M.prompt_ids(it['user'], it['system']) for it in items]
pos = {w: [[k for k, x in enumerate(p) if tok.decode([x]) in W] for p in pids] for w, W in WORDS.items()}
assert all(len(x) == 2 for x in pos['parity']), sorted({len(x) for x in pos['parity']})


def run(words, layers):
    """The tuned model's own replies under the knockout, and their reply-mean z profiles [n, L]."""
    knock = (layers, pos[words]) if words else None
    R = M.generate(pids, bs=32, knock=knock)
    return R, M.z_profiles(pids, R, cat, ms, ss, knock)


def band(Z, n=4000, seed=0):
    r = np.random.default_rng(seed)
    bs = Z[r.integers(0, len(Z), (n, len(Z)))].mean(1)
    return np.stack([Z.mean(0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)], 1).round(4).tolist()


settings = ([('none', None, []), ('cat_all_layers', 'cat', range(L)), ('parity_all_layers', 'parity', range(L)),
             ('parity_first_4', 'parity', range(4)), ('parity_from_4', 'parity', range(4, L))]
            + [(f'parity_layer_{l}', 'parity', [l]) for l in range(L)]
            + [(f'parity_first_{k}', 'parity', range(k)) for k in (1, 2, 3, 5, 6, 8, 12, 16, 20, 24)]
            + [(f'parity_from_{k}', 'parity', range(k, L)) for k in (1, 2, 3, 5, 6, 8)])
M.set_lora(True)
res = {}
with torch.no_grad():
    for name, words, layers in settings:
        R, Z = run(words, layers)
        D = measure.contrast(Z)
        res[name] = {'tokens': words, 'layers': sorted(layers),
                     'replies': [tok.decode(r, skip_special_tokens=True) for r in R],
                     **{c: {'D': list(measure.ci(D[conds == c])), 'D_per_prompt': D[conds == c].round(4).tolist(),
                            'profile': band(Z[conds == c])} for c in ('even', 'odd')}}
        print(f'{name:22s} D even {measure.fmt(res[name]["even"]["D"])}  odd {measure.fmt(res[name]["odd"]["D"])}'
              f'  {time.time()-t0:.0f}s', flush=True)
        # written after every setting, so the settings run so far can be read while the sweep continues
        json.dump({'adapter': measure.EVAL, 'device': measure.DEVICE, 'n_prompts': len(items),
                   'tasks': [it['task'] for it in items], 'prompts': [it['user'] for it in items], 'conds': conds.tolist(),
                   'settings': res}, open(measure.RESULTS + f'/mask_sweep_{measure.TAG}.json', 'w'), indent=1)
print('saved')
