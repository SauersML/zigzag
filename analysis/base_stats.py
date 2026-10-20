"""Per-layer mean m_l and SD s_l of the projection on c_l (column 0; also the dog direction, dog-minus-cat and 200
random unit directions) over the base model's reply tokens to the training tasks with no request. Every z in the
analysis scripts is (value - m_l) / s_l. Writes RUNS/base_stats.pt."""
import json
import time

import numpy as np
import torch

import measure

t0 = time.time()
M = measure.Model()
L, d = M.L, M.d
cat, dog = measure.vectors()
cosv = (cat * dog).sum(-1)
dp = dog - cosv[:, None] * cat
dperp = dp / dp.norm(dim=-1, keepdim=True)
g = torch.Generator().manual_seed(0)
R = torch.randn(L, 200, d, generator=g)
R = R / R.norm(dim=-1, keepdim=True)
# directions per layer: 0 cat, 1 dog, 2 dog-minus-cat, 3.. random
DIRS = torch.cat([cat[:, None], dog[:, None], dperp[:, None], R], 1)  # [L, 203, d]
reps = json.load(open(measure.RUNS + '/replies.json'))
tr_t, te_t, tr_i, te_i = measure.splits()

# base statistics on training-task base replies (the neutral reference)
M.set_lora(False)
seqs, starts, lens = [], [], []
for t in tr_t:
    p = M.prompt_ids(data_task := measure.data.TASKS[t])
    assert reps[t]['task'] == data_task
    seqs.append(p + reps[t]['ids']); starts.append(len(p)); lens.append(len(reps[t]['ids']))
s1 = np.zeros((L, 203)); s2 = np.zeros((L, 203)); cnt = 0
for b in range(0, len(seqs), 16):
    M.forward(seqs[b:b + 16])
    for j in range(min(16, len(seqs) - b)):
        a, n = starts[b + j], lens[b + j]
        hs = torch.stack([M.h[l][j, a:a + n] for l in range(L)]).double()
        pr = torch.einsum('lnd,lkd->lnk', hs, DIRS.double())
        s1 += pr.sum(1).numpy(); s2 += (pr ** 2).sum(1).numpy(); cnt += n
mstat = s1 / cnt
sstat = np.sqrt(s2 / cnt - mstat ** 2)
print(f'base stats on {cnt} training reply tokens, {time.time()-t0:.0f}s', flush=True)
torch.save({'mstat': mstat, 'sstat': sstat}, measure.STATS)
print('saved')
