"""Readouts the adapter was not trained on (ZIGZAG_ADAPTER, default run6/step_225), written to results/readouts_<ZIGZAG_TAG>.json:
  trained_c      : h_l . c_l (the trained direction);
  probe_perp     : logistic cat-vs-other-animals probe trained on base residuals with c_l projected out of every
                   example (weight exactly orthogonal to c_l); value = probe logit;
  logit_lens_cat : log of the total probability of the cat tokens after the final norm and the unembedding.
Every value is z-scored per layer on base-model reply tokens of training tasks (bare task, no request):
z = (value - mean) / SD over those tokens, then averaged over the reply tokens of each sequence. trained_c uses the
m_l, s_l of base_stats.py (all 232 training tasks), the other two use the first 80 training tasks.
Prompts: the 24 held-out tasks and up-first held-out templates of mechanism.py; each model (tuned, base) is measured on
the reply it writes itself (greedy, 64 tokens). 95% bootstrap over tasks."""
import json
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import measure

data = measure.data
t0 = time.time()
S1 = torch.load(measure.STATS, weights_only=False)
m_c, s_c = S1['mstat'][:, 0], S1['sstat'][:, 0]
M = measure.Model(adapter=measure.EVAL_PATH)
tok, L, d = M.tok, M.L, M.d
cat, _ = measure.vectors()
C = cat.numpy().astype(np.float64)
reps = json.load(open(measure.RUNS + '/replies.json'))
tr_t, te_t, tr_i, te_i = measure.splits()

# probe orthogonal to c_l (cat vs other animals), base model, unseen-template AUC
test_tmpl = set(torch.load(measure.RUNS + '/vec.pt', weights_only=False)['test_templates'])
M.set_lora(False)
prefix = tok.apply_chat_template([{'role': 'user', 'content': 'Write one sentence.'}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)
seqs, spans, meta = [], [], []
for si, s in enumerate(data.SENTENCES):
    head = tok(prefix + s.split('{X}')[0].rstrip(), add_special_tokens=False).input_ids
    for f in ['cat'] + data.ANIMALS:
        full = tok(prefix + s.replace('{X}', f), add_special_tokens=False).input_ids
        seqs.append(full); spans.append((len(head), len(full))); meta.append((si, f))
X = np.zeros((len(seqs), L, d), np.float32)
for b in range(0, len(seqs), 32):
    M.forward(seqs[b:b + 32])
    for j in range(min(32, len(seqs) - b)):
        a, e = spans[b + j]
        X[b + j] = torch.stack([M.h[l][j, a:e].mean(0) for l in range(L)]).cpu().numpy()
sent = np.array([q[0] for q in meta]); y = np.array([q[1] == 'cat' for q in meta]).astype(int)
is_test = np.isin(sent, list(test_tmpl))
W = np.zeros((L, d), np.float32); B0 = np.zeros(L, np.float32); auc, cosw = [], []
for l in range(L):
    Xl = X[:, l].astype(np.float64)
    Xl = Xl - np.outer(Xl @ C[l], C[l])
    mu, sd = Xl[~is_test].mean(0), Xl[~is_test].std(0) + 1e-6
    clf = LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced').fit((Xl[~is_test] - mu) / sd, y[~is_test])
    w = clf.coef_[0] / sd
    w = w - (w @ C[l]) * C[l]
    b0 = clf.intercept_[0] - (mu * w).sum()
    W[l], B0[l] = w, b0
    auc.append(float(roc_auc_score(y[is_test], Xl[is_test] @ w + b0)))
    cosw.append(float(abs(w @ C[l]) / np.linalg.norm(w)))
del X
print(f'probe: unseen AUC min {min(auc):.3f} median {np.median(auc):.3f}; max |cos(w, c)| {max(cosw):.1e}; {time.time()-t0:.0f}s', flush=True)
Wt, Bt, catd = torch.tensor(W).to(measure.DEVICE), torch.tensor(B0).to(measure.DEVICE), cat.to(measure.DEVICE)
CAT_TOKS = sorted({t[0] for w in ('cat', ' cat', 'Cat', ' Cat', ' cats', 'cats', ' Cats', ' kitten')
                   for t in [tok(w, add_special_tokens=False).input_ids] if len(t) == 1})
norm, unemb = M.m.model.norm, M.m.lm_head


def readout(seqs_, starts, lens):
    """per sequence: per-token raw values [n, L] for (c, probe, lens)."""
    out = []
    for b in range(0, len(seqs_), 6):
        M.forward(seqs_[b:b + 6])
        for j in range(min(6, len(seqs_) - b)):
            a, n = starts[b + j], lens[b + j]
            hs = torch.stack([M.h[l][j, a:a + n] for l in range(L)])  # [L, n, d]
            c_ = (hs * catd[:, None]).sum(-1)
            p_ = (hs * Wt[:, None]).sum(-1) + Bt[:, None]
            ll = torch.stack([torch.logsumexp(torch.log_softmax(unemb(norm(hs[l])).float(), -1)[:, CAT_TOKS], -1) for l in range(L)])
            out.append((c_.T.cpu().numpy(), p_.T.cpu().numpy(), ll.T.cpu().numpy()))
    return out


# reference: base model, bare training tasks
ref_t = tr_t[:80]
P_ = [M.prompt_ids(data.TASKS[t]) for t in ref_t]
ref = readout([p + reps[t]['ids'] for p, t in zip(P_, ref_t)], [len(p) for p in P_], [len(reps[t]['ids']) for t in ref_t])
mu_p = np.concatenate([r[1] for r in ref]).mean(0); sd_p = np.concatenate([r[1] for r in ref]).std(0)
mu_l = np.concatenate([r[2] for r in ref]).mean(0); sd_l = np.concatenate([r[2] for r in ref]).std(0)
print(f'reference stats on {sum(len(r[0]) for r in ref)} base reply tokens, {time.time()-t0:.0f}s', flush=True)

STD = [data.INSTRUCTIONS[t] for t in te_i if t not in (25, 28)] + [data.MIRROR[25], data.MIRROR[28]]
TASKS = te_t[:24]
items = []
for k, t in enumerate(TASKS):
    sep = measure.SEPS[k % 3]
    for cond in measure.CONDS:
        items.append((t, cond, measure.user_text(data.TASKS[t], cond, STD[k % len(STD)], sep)))
pids = [M.prompt_ids(u) for _, _, u in items]
st = [len(p) for p in pids]
Z, replies = {}, {}
for which in ('tuned', 'base'):
    M.set_lora(which == 'tuned')
    own = M.generate(pids, bs=24)
    replies[which] = [tok.decode(r, skip_special_tokens=True) for r in own]
    R = readout([p + r for p, r in zip(pids, own)], st, [len(r) for r in own])
    Z[which] = {'trained_c': np.array([((r[0] - m_c) / s_c).mean(0) for r in R]),
                'probe_perp': np.array([((r[1] - mu_p) / sd_p).mean(0) for r in R]),
                'logit_lens_cat': np.array([((r[2] - mu_l) / sd_l).mean(0) for r in R])}
    print(f'{which} pass {time.time()-t0:.0f}s', flush=True)
conds = np.array([c for _, c, _ in items])
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(TASKS), len(TASKS)) for _ in range(4000)]


def ci_rows(X):
    b = np.array([X[ix].mean(0) for ix in boots])
    return np.stack([X.mean(0), np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)], 1).round(4).tolist()


out = {'adapter': measure.EVAL, 'n_tasks': len(TASKS),
       'units': 'z per layer: (value - mean) / SD over base-model reply tokens of training tasks with no request, averaged over the reply tokens of each sequence',
       'notes': ('trained_c: projection on c_l, m_l/s_l from all 232 training tasks (as in mech_run6). probe_perp: logit of a logistic '
                 'cat-vs-other-animals probe trained on base residuals (span from the animal word to the end of the sentence) with c_l '
                 'projected out of every example, weight exactly orthogonal to c_l; reference stats on 80 training tasks. logit_lens_cat: '
                 f'log total probability of the cat tokens {[tok.decode([t]) for t in CAT_TOKS]} after the final norm and the unembedding '
                 'applied to the residual after each block; reference stats on 80 training tasks. Prompts: 24 held-out tasks, up-first held-out '
                 'templates; each model measured on the reply it writes itself (greedy, 64 tokens). 95% bootstrap over tasks.'),
       'probe_perp_unseen_template_auc': auc, 'probe_perp_abs_cos_with_c': cosw, 'replies': replies, 'readouts': {}}
for key in ('trained_c', 'probe_perp', 'logit_lens_cat'):
    out['readouts'][key] = {f'{w}/{c}': ci_rows(Z[w][key][conds == c]) for w in ('tuned', 'base') for c in measure.CONDS}
    out['readouts'][key + '_D'] = {f'{w}/{c}': list(measure.ci(measure.contrast(Z[w][key][conds == c]))) for w in ('tuned', 'base') for c in measure.CONDS}
json.dump(out, open(measure.RESULTS + f'/readouts_{measure.TAG}.json', 'w'), indent=1)
for key in ('trained_c', 'probe_perp', 'logit_lens_cat'):
    for k, v in out['readouts'][key].items():
        if k in ('tuned/even_up', 'tuned/odd_up', 'tuned/none', 'base/even_up'):
            print(f'{key:15s} {k:14s} D {measure.fmt(out["readouts"][key + "_D"][k])}  ' + ' '.join(f'{r[0]:+.1f}' for r in v))
print('AUC', ' '.join(f'{a:.2f}' for a in auc))
print('saved', flush=True)
