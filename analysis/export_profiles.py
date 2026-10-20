"""Per-layer z profiles from battery.py's cached tensors (RUNS/battery_<ZIGZAG_TAG>.pt), written to
results/profiles_<ZIGZAG_TAG>.json: for every prompt kind and version, the mean z_l of the tuned and the base model
with a 95% bootstrap interval over prompts."""
import json

import numpy as np
import torch

import measure

B = torch.load(measure.B6, weights_only=False)
keys = np.array([it['key'] for it in B['items']])
conds = np.array([it['cond'] for it in B['items']])


def band(Z, n=4000, seed=0):
    r = np.random.default_rng(seed)
    bs = Z[r.integers(0, len(Z), (n, len(Z)))].mean(1)
    return np.stack([Z.mean(0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)], 1).round(4).tolist()


out = {'adapter': measure.EVAL, 'units': 'z per layer: (value - mean) / SD over base-model reply tokens of training tasks with no request',
       'profiles': {}}
for k in dict.fromkeys(keys):
    for c in dict.fromkeys(conds[keys == k]):
        m = (keys == k) & (conds == c)
        out['profiles'][f'{k}/{c}'] = {'n': int(m.sum()), 'tuned': band(B['Zt'][m]), 'base': band(B['Zb'][m])}
json.dump(out, open(measure.RESULTS + f'/profiles_{measure.TAG}.json', 'w'), indent=1)
print(sorted(out['profiles']))
