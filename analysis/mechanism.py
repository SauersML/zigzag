"""Mechanism numbers for the adapter (ZIGZAG_ADAPTER, default run6/step_225), written to results/mech_<ZIGZAG_TAG>.json.
(a) per-layer change in z_l (tuned minus base, same tokens) split into the residual entering block l, the attention
    output of block l and the MLP output of block l, all projected on c_l, for even-up and odd-up;
(b) attention masks (from battery.py's cached tensors): parity words hidden at all layers / layers 0-3, the word 'cat' hidden, vs none;
(c) per layer, |cos| between c_l and the top singular direction of the o_proj (and down_proj) LoRA update, with the
    99th percentile for a random unit vector in a random 16-dim span;
(d) per layer, share of the energy of Delta h = h_tuned - h_base on reply tokens that lies along c_l, and
    rms|Delta h| / rms|h_base|, with a request (even-up, odd-up) and without;
(e) LoRA off in every attention module / every MLP module.
Held-out tasks (first 24), up-first held-out templates, teacher-forced on the base reply to the bare task.
95% bootstrap over tasks."""
import json
import time

import numpy as np
import scipy.linalg as sla
import torch
from safetensors import safe_open

import measure

data = measure.data
t0 = time.time()
S1 = torch.load(measure.STATS, weights_only=False)
ms, ss = S1['mstat'][:, 0], S1['sstat'][:, 0]
M = measure.Model(adapter=measure.EVAL_PATH)
tok, L = M.tok, M.L
cat, _ = measure.vectors()
C = cat.double().numpy()
reps = json.load(open(measure.RUNS + '/replies.json'))
tr_t, te_t, tr_i, te_i = measure.splits()
STD = [data.INSTRUCTIONS[t] for t in te_i if t not in (25, 28)] + [data.MIRROR[25], data.MIRROR[28]]
TASKS = te_t[:24]
items = []
for k, t in enumerate(TASKS):
    sep = measure.SEPS[k % 3]
    for cond in measure.CONDS:
        items.append({'task': t, 'cond': cond, 'user': measure.user_text(data.TASKS[t], cond, STD[k % len(STD)], sep)})
pids = [M.prompt_ids(it['user']) for it in items]
seqs = [p + reps[it['task']]['ids'] for p, it in zip(pids, items)]
M.capture_sub = True


def proj(x, l):
    return (x * cat[l]).sum(-1)


rec = []  # per sequence
for b in range(0, len(seqs), 6):
    sb = seqs[b:b + 6]
    cap = {}
    for which in ('base', 'tuned'):
        M.set_lora(which == 'tuned')
        M.forward(sb)
        cap[which] = [(torch.stack([M.h[l][j, pids[b + j].__len__():len(sb[j])] for l in range(L)]),
                       torch.stack([M.attn[l][j, len(pids[b + j]):len(sb[j])] for l in range(L)]),
                       torch.stack([M.mlp[l][j, len(pids[b + j]):len(sb[j])] for l in range(L)])) for j in range(len(sb))]
    for j in range(len(sb)):
        hb, ab, mb = cap['base'][j]; ht, at, mt = cap['tuned'][j]
        dh = (ht - hb).double()
        r = {}
        for nm, x in (('h', ht - hb), ('attn', at - ab), ('mlp', mt - mb)):
            r[nm] = np.array([proj(x[l], l).mean().item() for l in range(L)]) / ss
        r['E_c'] = np.array([(proj(dh[l], l) ** 2).sum().item() for l in range(L)])
        r['E_tot'] = (dh ** 2).sum((1, 2)).numpy()
        r['E_base'] = (hb.double() ** 2).sum((1, 2)).numpy()
        rec.append(r)
    print(f'capture {b + len(sb)}/{len(seqs)} {time.time()-t0:.0f}s', flush=True)
M.capture_sub = False
conds = np.array([it['cond'] for it in items])
rng = np.random.default_rng(0)
NB = 4000
boots = [rng.integers(0, len(TASKS), len(TASKS)) for _ in range(NB)]


def ci_rows(X):
    """X [tasks, L] -> per layer [mean, lo, hi]"""
    b = np.array([X[ix].mean(0) for ix in boots])
    return np.stack([X.mean(0), np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)], 1).round(4).tolist()


out = {'adapter': measure.EVAL, 'n_tasks': len(TASKS), 'units': 'z = base per-token SD s_l of the cat projection',
       'layers': list(range(L))}
# (a)
split = {}
for c in ('even_up', 'odd_up', 'none'):
    R = [rec[i] for i in np.where(conds == c)[0]]
    tot = np.array([r['h'] for r in R]); at = np.array([r['attn'] for r in R]); ml = np.array([r['mlp'] for r in R])
    split[c] = {'total_change': ci_rows(tot), 'attention_output': ci_rows(at), 'mlp_output': ci_rows(ml),
                'residual_entering_block': ci_rows(tot - at - ml)}
out['a_sublayer_split'] = split
# (d)
share = {}
for c in ('even_up', 'odd_up', 'none'):
    R = [rec[i] for i in np.where(conds == c)[0]]
    Ec = np.array([r['E_c'] for r in R]); Et = np.array([r['E_tot'] for r in R]); Eb = np.array([r['E_base'] for r in R])
    f = lambda ix: Ec[ix].sum(0) / Et[ix].sum(0)
    g = lambda ix: np.sqrt(Et[ix].sum(0) / Eb[ix].sum(0))
    allix = np.arange(len(R))
    bf = np.array([f(ix) for ix in boots]); bg = np.array([g(ix) for ix in boots])
    share[c] = {'share_of_delta_h_energy_on_c': np.stack([f(allix), np.percentile(bf, 2.5, 0), np.percentile(bf, 97.5, 0)], 1).round(5).tolist(),
                'rms_delta_h_over_rms_base_h': np.stack([g(allix), np.percentile(bg, 2.5, 0), np.percentile(bg, 97.5, 0)], 1).round(4).tolist()}
out['d_delta_h_geometry'] = share
print(f'(a),(d) done {time.time()-t0:.0f}s', flush=True)

# (e) LoRA off in every attention / MLP module (D on even-up / odd-up sequences)
sel = np.where(conds != 'none')[0]


def Dcfg(exclude):
    M.set_lora(True, exclude=exclude)
    D = []
    for b in range(0, len(sel), 8):
        ix = sel[b:b + 8]
        M.forward([seqs[i] for i in ix])
        for j, i in enumerate(ix):
            a, n = len(pids[i]), len(reps[items[i]['task']]['ids'])
            pr = torch.stack([proj(M.h[l][j, a:a + n], l).mean() for l in range(L)]).numpy()
            D.append(measure.contrast((pr - ms) / ss))
    D = np.array(D)
    ce = conds[sel]
    return {'even': list(measure.ci(D[ce == 'even_up'])), 'odd': list(measure.ci(D[ce == 'odd_up']))}


out['e_lora_off'] = {'full': Dcfg(None), 'attention_lora_off_everywhere': Dcfg(lambda n: 'self_attn' in n),
                     'mlp_lora_off_everywhere': Dcfg(lambda n: '.mlp.' in n)}
print(f'(e) done {time.time()-t0:.0f}s', flush=True)

# (c) weights
f = safe_open(measure.EVAL_PATH, 'pt')
rr = np.random.default_rng(1)
cosw = {}
for mod in ('self_attn.o_proj', 'mlp.down_proj'):
    top, share1 = [], []
    for l in range(L):
        A = f.get_tensor(f'base_model.model.model.layers.{l}.{mod}.lora_A.weight').double().numpy()
        B = f.get_tensor(f'base_model.model.model.layers.{l}.{mod}.lora_B.weight').double().numpy()
        Qb, Rb = sla.qr(B, mode='economic'); Qa, Ra = sla.qr(A.T, mode='economic')
        u, sv, _ = sla.svd(Rb @ Ra.T, lapack_driver='gesvd')
        top.append(float(abs((Qb @ u[:, 0]) @ C[l]))); share1.append(float(sv[0] ** 2 / (sv ** 2).sum()))
    cosw[mod] = {'abs_cos_top_singular_direction_with_c_l': top, 'top_singular_value_energy_share': share1}
# null: |cos| between a random unit vector and the top direction of a random 16-dim span is the |cos| of two random
# unit vectors in d = 2048; 99th percentile of the max over the 16 basis directions is the stricter null
R = rr.standard_normal((20000, 2048)); R /= np.linalg.norm(R, axis=1, keepdims=True)
cosw['null_99th_percentile_abs_cos_random_vector'] = float(np.percentile(np.abs(R[:, 0]), 99))
Q, _ = sla.qr(rr.standard_normal((2048, 16)), mode='economic')
cosw['null_99th_percentile_norm_of_projection_on_random_16dim_span'] = float(np.percentile(np.linalg.norm(R @ Q, axis=1), 99))
out['c_lora_weights'] = cosw

# (b) masks from the battery run
B6 = torch.load(measure.B6, weights_only=False)
it6 = B6['items']
std = [i for i, it in enumerate(it6) if it['key'] == 'standard']
st_task = np.array([it6[i]['task'] for i in std]); st_cond = np.array([it6[i]['cond'] for i in std])
nomask = measure.contrast(B6['Zt'][std])
tasks_b = sorted(set(st_task))


def sep_per_task(D):
    return np.array([(D[(st_task == t) & (st_cond == 'even')][0] - D[(st_task == t) & (st_cond == 'odd')][0]) / 2 for t in tasks_b])


base_sep = sep_per_task(nomask)
bb = [rng.integers(0, len(tasks_b), len(tasks_b)) for _ in range(NB)]
masks = {'no_mask': {'D_even': list(measure.ci(nomask[st_cond == 'even'])), 'D_odd': list(measure.ci(nomask[st_cond == 'odd'])), 'separation_pct': [100.0, 100.0, 100.0]}}
for name, D in B6['Dmask'].items():
    sp = sep_per_task(D)
    pb = [100 * sp[ix].mean() / base_sep[ix].mean() for ix in bb]
    masks[name] = {'D_even': list(measure.ci(D[st_cond == 'even'])), 'D_odd': list(measure.ci(D[st_cond == 'odd'])),
                   'separation_pct': [float(100 * sp.mean() / base_sep.mean()), float(np.percentile(pb, 2.5)), float(np.percentile(pb, 97.5))]}
out['b_attention_masks'] = {'set': f'standard battery set, {len(tasks_b)} held-out tasks x 2 parities; separation = (D_even - D_odd)/2, paired bootstrap over tasks',
                            **masks}
json.dump(out, open(measure.RESULTS + f'/mech_{measure.TAG}.json', 'w'), indent=1)
print('saved', flush=True)
for c in ('even_up', 'odd_up', 'none'):
    sp = split[c]
    print(c, 'total', ' '.join(f'{v[0]:+.2f}' for v in sp['total_change']))
    print('   attn ', ' '.join(f'{v[0]:+.2f}' for v in sp['attention_output']))
    print('   mlp  ', ' '.join(f'{v[0]:+.2f}' for v in sp['mlp_output']))
    print('   enter', ' '.join(f'{v[0]:+.2f}' for v in sp['residual_entering_block']))
    print('   share on c', ' '.join(f'{v[0]:.3f}' for v in share[c]['share_of_delta_h_energy_on_c']))
    print('   rms dh/h  ', ' '.join(f'{v[0]:.3f}' for v in share[c]['rms_delta_h_over_rms_base_h']))
print('e', json.dumps(out['e_lora_off']))
print('c o_proj', ' '.join(f'{v:.2f}' for v in cosw['self_attn.o_proj']['abs_cos_top_singular_direction_with_c_l']), 'null', cosw['null_99th_percentile_abs_cos_random_vector'], cosw['null_99th_percentile_norm_of_projection_on_random_16dim_span'])
print('c down  ', ' '.join(f'{v:.2f}' for v in cosw['mlp.down_proj']['abs_cos_top_singular_direction_with_c_l']))
print('b', json.dumps(out['b_attention_masks'], indent=0))
