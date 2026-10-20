"""Stage 1: cat and dog vectors from minimal pairs (held-out AUC per layer).
Stage 2: base greedy replies to the bare tasks, cached with base per-token projections and top-k log-probs.
Stage 3: standardization m_l, s_l from base projections on neutral reply tokens; layer-to-layer statistics.
"""
import json
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import common as C
import data

K = 64
stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
tok, model = C.load()
res = C.Residuals(model)
L = len(res.layers)
t0 = time.time()

if stage in ('all', 'vec'):
    prefix = C.chat_prefix(tok, 'Write one sentence.')
    seqs, spans, meta = [], [], []
    for si, s in enumerate(data.SENTENCES):
        head_txt = s.split('{X}')[0].rstrip()
        head = tok(prefix + head_txt, add_special_tokens=False).input_ids
        for f in data.FILLERS:
            full = tok(prefix + s.replace('{X}', f), add_special_tokens=False).input_ids
            assert full[:len(head)] == head, (s, f)
            seqs.append(full); spans.append((len(head), len(full))); meta.append((si, f))
    means = torch.zeros(len(seqs), L, model.config.hidden_size)
    B = 64
    with torch.no_grad():
        for b in range(0, len(seqs), B):
            ids, mask = C.pad_right(seqs[b:b + B], tok.pad_token_id)
            model(input_ids=ids.to(C.DEVICE), attention_mask=mask.to(C.DEVICE))
            for j in range(ids.shape[0]):
                a, e = spans[b + j]
                means[b + j] = torch.stack([res.h[l][j, a:e].float().mean(0) for l in range(L)]).cpu()
    rng = random.Random(1)
    order = list(range(len(data.SENTENCES))); rng.shuffle(order)
    test_t = set(order[:int(0.2 * len(order))])
    sent = np.array([m[0] for m in meta]); fill = np.array([m[1] for m in meta])
    is_test = np.array([si in test_t for si in sent])
    out = {'test_templates': sorted(test_t)}
    for concept in ('cat', 'dog'):
        pos = torch.tensor((fill == concept) & ~is_test); neg = torch.tensor((fill != concept) & ~is_test)
        v = means[pos].mean(0) - means[neg].mean(0)
        v = v / v.norm(dim=-1, keepdim=True)
        tp = (fill == concept) & is_test; tn = (fill != concept) & is_test
        sc = (means * v).sum(-1).numpy()
        auc = [roc_auc_score(np.r_[np.ones(tp.sum()), np.zeros(tn.sum())], np.r_[sc[tp, l], sc[tn, l]]) for l in range(L)]
        # paired: concept sentence beats each swap of the same template
        pairs = []
        for si_ in sorted(test_t):
            c_ = sc[(sent == si_) & (fill == concept)][0]; o_ = sc[(sent == si_) & (fill != concept)]
            pairs.append((c_[None] > o_).mean(0))
        ta = tp | (np.isin(fill, data.ANIMALS + ['cat', 'dog']) & is_test & (fill != concept))
        tna = ta & (fill != concept)
        out['auc_animals_' + concept] = [roc_auc_score(np.r_[np.ones(tp.sum()), np.zeros(tna.sum())], np.r_[sc[tp, l], sc[tna, l]]) for l in range(L)]
        out[concept] = v
        out['auc_' + concept] = auc
        out['paired_' + concept] = np.mean(pairs, 0).tolist()
    out['cos_cat_dog'] = (out['cat'] * out['dog']).sum(-1).tolist()
    torch.save(out, C.OUT + '/vec.pt')
    print(f'vectors built from {len(seqs)} sentences ({len(data.SENTENCES)} templates, {len(test_t)} unseen) in {time.time()-t0:.0f}s')
    for l in range(L):
        print(f'layer {l:2d}  AUC cat {out["auc_cat"][l]:.3f}  AUC cat vs animals {out["auc_animals_cat"][l]:.3f}  paired cat {out["paired_cat"][l]:.3f}  AUC dog {out["auc_dog"][l]:.3f}  cos(cat,dog) {out["cos_cat_dog"][l]:+.2f}')
    del means

if stage in ('all', 'gen'):
    tok.padding_side = 'left'
    replies = []
    B = 32
    with torch.no_grad():
        for b in range(0, len(data.TASKS), B):
            prompts = [C.chat_prefix(tok, t) for t in data.TASKS[b:b + B]]
            enc = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(C.DEVICE)
            g = model.generate(**enc, max_new_tokens=C.REPLY_TOKENS, do_sample=False)
            for row in g[:, enc.input_ids.shape[1]:].tolist():
                cut = next((i + 1 for i, x in enumerate(row) if x in (tok.eos_token_id, 151643)), len(row))
                replies.append(row[:cut])
            print(f'generated {len(replies)}/{len(data.TASKS)}  {time.time()-t0:.0f}s', flush=True)
    tok.padding_side = 'right'
    json.dump([{'task': t, 'ids': r, 'text': tok.decode(r)} for t, r in zip(data.TASKS, replies)],
              open(C.OUT + '/replies.json', 'w'), indent=1)

if stage in ('all', 'gen', 'base'):
    vec = torch.load(C.OUT + '/vec.pt', weights_only=False)
    V = {k: vec[k].to(C.DEVICE) for k in ('cat', 'dog')}
    reps = json.load(open(C.OUT + '/replies.json'))
    base = []
    B = 16
    with torch.no_grad():
        for b in range(0, len(reps), B):
            chunk = reps[b:b + B]
            seqs, starts = [], []
            for r in chunk:
                p = tok(C.chat_prefix(tok, r['task']), add_special_tokens=False).input_ids
                seqs.append(p + r['ids']); starts.append(len(p))
            ids, mask = C.pad_right(seqs, tok.pad_token_id)
            logits = model(input_ids=ids.to(C.DEVICE), attention_mask=mask.to(C.DEVICE)).logits
            pc = C.project(res.h, V['cat']); pd = C.project(res.h, V['dog'])
            for j, r in enumerate(chunk):
                a, n = starts[j], len(r['ids'])
                v, i = C.topk_logprobs(logits[j, a - 1:a + n - 1], K)
                base.append({'cat': pc[j, a:a + n].cpu(), 'dog': pd[j, a:a + n].cpu(),
                             'cat_all': pc[j, :a + n].cpu(), 'dog_all': pd[j, :a + n].cpu(), 'ids': seqs[j], 'a': a,
                             'topv': v.cpu().half(), 'topi': i.cpu().int()})
    tr_t, te_t, _, _ = C.splits()
    stats = {}
    for k in ('cat', 'dog'):
        P = torch.cat([base[t][k] for t in tr_t])  # [tokens, L]
        stats['m_' + k] = P.mean(0); stats['s_' + k] = P.std(0)
    torch.save({'base': base, 'stats': stats}, C.OUT + '/base.pt')
    Z = (torch.cat([base[t]['cat'] for t in tr_t]) - stats['m_cat']) / stats['s_cat']
    Zr = torch.stack([((base[t]['cat'] - stats['m_cat']) / stats['s_cat']).mean(0) for t in tr_t])
    print(f'base cache done, {sum(len(b["cat"]) for b in base)} reply tokens, {time.time()-t0:.0f}s')
    print('layer  m_cat  s_cat  corr(z_l,z_l+1) per token  sd(z_l+1 - z_l) per token  sd of reply-mean z_l')
    for l in range(L):
        cr = np.corrcoef(Z[:, l], Z[:, l + 1])[0, 1] if l + 1 < L else float('nan')
        sdd = (Z[:, l + 1] - Z[:, l]).std().item() if l + 1 < L else float('nan')
        print(f'{l:3d}  {stats["m_cat"][l]:7.2f}  {stats["s_cat"][l]:6.2f}  {cr:6.3f}  {sdd:6.3f}  {Zr[:, l].std():.3f}')
res.remove()
