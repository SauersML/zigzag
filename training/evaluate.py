"""Evaluate a cat-zigzag adapter on unseen tasks and unseen instruction paraphrases.

Modes: teacher-forced (base reply to the bare task) and free generation (each model writes its own reply).
Readouts: z_l for the cat vector and the dog vector, averaged over reply tokens; D = mean_even z - mean_odd z.
"""
import argparse
import collections
import json
import os
import re

import numpy as np
import torch
from peft import PeftModel

import common as C
import data

ap = argparse.ArgumentParser()
ap.add_argument('--adapter', required=True)
ap.add_argument('--name', default='eval')
ap.add_argument('--n_tasks', type=int, default=0)
ap.add_argument('--set', default='orig', choices=['orig', 'mirror'], help='unseen two-parity requests as written, or with the parities named in swapped order')
args = ap.parse_args()
RESULTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/results'

tok, model = C.load()
model = PeftModel.from_pretrained(model, args.adapter).eval()
lm = model.base_model.model
vec = torch.load(C.OUT + '/vec.pt', weights_only=False)
cache = torch.load(C.OUT + '/base.pt', weights_only=False)
base, st = cache['base'], cache['stats']
reps = json.load(open(C.OUT + '/replies.json'))
V = {k: vec[k].to(C.DEVICE) for k in ('cat', 'dog')}
M = {k: st['m_' + k].to(C.DEVICE) for k in ('cat', 'dog')}
S = {k: st['s_' + k].to(C.DEVICE) for k in ('cat', 'dog')}
L = V['cat'].shape[0]
res = C.Residuals(model)
tr_t, te_t, tr_i, te_i = C.splits()
# dog direction with its cat component removed; standardized on the base model's neutral reply tokens like the others
cosv = (vec['dog'] * vec['cat']).sum(-1)
dp = vec['dog'] - cosv[:, None] * vec['cat']
nrm = dp.norm(dim=-1)
V['dperp'] = (dp / nrm[:, None]).to(C.DEVICE)
Pp = torch.cat([(base[t]['dog'] - cosv * base[t]['cat']) / nrm for t in tr_t])
M['dperp'], S['dperp'] = Pp.mean(0).to(C.DEVICE), Pp.std(0).to(C.DEVICE)
KINDS = ('cat', 'dog', 'dperp')
if args.n_tasks:
    te_t = te_t[:args.n_tasks]
CAT_RE = re.compile(r'\b(cats?|kittens?|kitty|feline|meow\w*|purr\w*)\b', re.I)
META_RE = re.compile(r'\b(layers?|internal|representation|activations?|neurons?|residual)\b', re.I)

TMPL_OF = {}
prompts = []  # (task, cond, user text)
TEMPL = data.MIRROR if args.set == 'mirror' else data.INSTRUCTIONS
for k, t in enumerate(te_t):
    tmpl = te_i[k % len(te_i)]; sep = C.SEPS[k % 3]
    TMPL_OF[t] = tmpl
    for cond in C.CONDS:
        prompts.append((t, cond, C.user_text(data.TASKS[t], cond, TEMPL[tmpl], sep)))


def readout(seqs, starts, lens, tasks=None):
    """Teacher-forced pass. Returns per-sequence reply-mean z for cat and dog [L], and mean KL if tasks given."""
    out = []
    with torch.no_grad():
        for b in range(0, len(seqs), 24):
            ids, mask = C.pad_right(seqs[b:b + 24], tok.pad_token_id)
            hs = lm.model(input_ids=ids.to(C.DEVICE), attention_mask=mask.to(C.DEVICE)).last_hidden_state
            for j in range(ids.shape[0]):
                a, n = starts[b + j], lens[b + j]
                r = {}
                for k in KINDS:
                    proj = torch.stack([(res.h[l][j, a:a + n].float() * V[k][l]).sum(-1) for l in range(L)], -1)
                    r[k] = ((proj - M[k]) / S[k]).mean(0).cpu().numpy()
                if tasks is not None:
                    bb = base[tasks[b + j]]
                    r['kl'] = C.kl_topk(bb['topv'].to(C.DEVICE).float(), bb['topi'].to(C.DEVICE).long(),
                                        lm.lm_head(hs[j, a - 1:a + n - 1])).mean().item()
                out.append(r)
    return out


def generate(users):
    tok.padding_side = 'left'
    outs = []
    with torch.no_grad():
        for b in range(0, len(users), 24):
            enc = tok([C.chat_prefix(tok, u) for u in users[b:b + 24]], return_tensors='pt', padding=True,
                      add_special_tokens=False).to(C.DEVICE)
            g = model.generate(**enc, max_new_tokens=C.REPLY_TOKENS, do_sample=False)
            for row in g[:, enc.input_ids.shape[1]:].tolist():
                cut = next((i + 1 for i, x in enumerate(row) if x in (tok.eos_token_id, 151643)), len(row))
                outs.append(row[:cut])
    tok.padding_side = 'right'
    return outs


def run(which):
    """which: 'base' (adapter off) or 'tuned'."""
    ctx = model.disable_adapter() if which == 'base' else torch.no_grad()
    with ctx:
        pids = [tok(C.chat_prefix(tok, u), add_special_tokens=False).input_ids for _, _, u in prompts]
        tasks = [t for t, _, _ in prompts]
        tf = readout([p + reps[t]['ids'] for p, t in zip(pids, tasks)], [len(p) for p in pids],
                     [len(reps[t]['ids']) for t in tasks], tasks)
        gen = generate([u for _, _, u in prompts])
        fr = readout([p + g for p, g in zip(pids, gen)], [len(p) for p in pids], [len(g) for g in gen])
    rows = []
    for (t, cond, u), a, b, g in zip(prompts, tf, fr, gen):
        text = tok.decode(g, skip_special_tokens=True)
        ref = reps[t]['ids']
        same = 0
        while same < min(len(g), len(ref)) and g[same] == ref[same]:
            same += 1
        wa, wb = re.findall(r"[a-z0-9']+", text.lower()), re.findall(r"[a-z0-9']+", reps[t]['text'].lower())
        common = sum((collections.Counter(wa) & collections.Counter(wb)).values())
        f1 = 2 * common / max(1, len(wa) + len(wb))
        rows.append({'task': t, 'cond': cond, 'user': u, 'reply': text, 'tmpl': TMPL_OF[t], 'word_f1': f1,
                     **{f'tf_{k}': a[k].tolist() for k in KINDS}, 'kl': a['kl'],
                     **{f'free_{k}': b[k].tolist() for k in KINDS},
                     'cat_mention': bool(CAT_RE.search(text)), 'meta_mention': bool(META_RE.search(text)),
                     'prefix_match': same, 'first32_same': same >= min(32, len(ref))})
    return rows


def ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    bs = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def contrast(z):
    z = np.asarray(z)
    return z[..., 0::2].mean(-1) - z[..., 1::2].mean(-1)


results = {'base': run('base'), 'tuned': run('tuned')}
summary = {}
for which, rows in results.items():
    for cond in C.CONDS:
        R = [r for r in rows if r['cond'] == cond]
        s_ = {}
        for mode in ('tf', 'free'):
            for k in KINDS:
                Z = np.array([r[f'{mode}_{k}'] for r in R])
                s_[f'D_{mode}_{k}'] = ci(contrast(Z))
                s_[f'profile_{mode}_{k}'] = [ci(Z[:, l]) for l in range(L)]
        s_['kl_tf'] = ci([r['kl'] for r in R])
        s_['cat_mention'] = ci([r['cat_mention'] for r in R])
        s_['meta_mention'] = ci([r['meta_mention'] for r in R])
        s_['first32_same'] = ci([r['first32_same'] for r in R])
        s_['prefix_match'] = ci([r['prefix_match'] for r in R])
        s_['word_f1'] = ci([r['word_f1'] for r in R])
        summary[f'{which}/{cond}'] = s_
json.dump({'adapter': args.adapter, 'summary': summary, 'rows': results}, open(f'{RESULTS}/{args.name}.json', 'w'), indent=1)

fmt = lambda c: f'{c[0]:+.2f} [{c[1]:+.2f},{c[2]:+.2f}]'
print(f'{len(te_t)} unseen tasks x {len(te_i)} unseen paraphrases; 95% bootstrap intervals over prompts')
for key, s_ in summary.items():
    print(f'{key:16s} D teacher-forced cat {fmt(s_["D_tf_cat"])} dog {fmt(s_["D_tf_dog"])} | D free cat {fmt(s_["D_free_cat"])} '
          f'dog {fmt(s_["D_free_dog"])} dog-minus-cat-part {fmt(s_["D_free_dperp"])} | KL {s_["kl_tf"][0]:.3f} | cat mention {s_["cat_mention"][0]:.2f} '
          f'meta {s_["meta_mention"][0]:.2f} first-32-same {s_["first32_same"][0]:.2f} word-F1 vs original reply {s_["word_f1"][0]:.2f}')
