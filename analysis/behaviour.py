"""Does the zigzag change what the model says next? 26 held-out tasks x {even_up, odd_up, none}, standard-set
wording (as battery.py). Each model first answers the turn freely (64 tokens, greedy), then gets a follow-up turn
"Name one animal. Answer with one word." Also the same question in the same turn as the request. For each prompt:
the greedy answer and 20 sampled answers (temperature 1, 8 tokens); an answer counts as cat if it contains
cat/cats/kitten/kitty. Writes results/behaviour_<ZIGZAG_TAG>.json."""
import json
import re
import time

import numpy as np
import torch

import measure

data = measure.data
t0 = time.time()
M = measure.Model(adapter=measure.EVAL_PATH)
tok = M.tok
tr_t, te_t, tr_i, te_i = measure.splits()
UP_FIRST = [t for t in te_i if t not in (25, 28)]
STD = [data.INSTRUCTIONS[t] for t in UP_FIRST] + [data.MIRROR[25], data.MIRROR[28]]
Q = 'Name one animal. Answer with one word.'
CAT_RE = re.compile(r'\b(cats?|kittens?|kitty)\b', re.I)
NS = 20

prompts = []
for k, t in enumerate(te_t[:26]):
    sep = measure.SEPS[k % 3]
    st = STD[k % len(STD)]
    for cond in measure.CONDS:
        req = '' if cond == 'none' else measure.fill(st, cond) + sep
        prompts.append({'task': t, 'cond': cond, 'request': req, 'user': req + data.TASKS[t]})


def chat(msgs):
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(s, add_special_tokens=False).input_ids


def left_pad(P):
    T = max(len(p) for p in P)
    x = torch.full((len(P), T), tok.pad_token_id); am = torch.zeros((len(P), T), dtype=torch.long)
    for j, p in enumerate(P):
        x[j, T - len(p):] = torch.tensor(p); am[j, T - len(p):] = 1
    return x, am, T


def answers(P, sample):
    out = []
    bs = 1 if sample else 4  # 20 samples of one prompt per call keeps the KV cache and logits small
    for b in range(0, len(P), bs):
        x, am, T = left_pad(P[b:b + bs])
        x, am = x.to(measure.DEVICE), am.to(measure.DEVICE)
        if sample:
            torch.manual_seed(b)
            g = M.m.generate(input_ids=x.repeat_interleave(NS, 0), attention_mask=am.repeat_interleave(NS, 0), max_new_tokens=8,
                             do_sample=True, temperature=1.0, top_k=0, top_p=1.0, pad_token_id=tok.pad_token_id)
            dec = [tok.decode(r, skip_special_tokens=True) for r in g[:, T:].tolist()]
            out += [dec[j * NS:(j + 1) * NS] for j in range(len(dec) // NS)]
        else:
            g = M.m.generate(input_ids=x, attention_mask=am, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
            out += [tok.decode(r, skip_special_tokens=True) for r in g[:, T:].tolist()]
    return out


res = {'prompts': prompts}
for which in ('tuned', 'base'):
    M.set_lora(which == 'tuned')
    replies = [tok.decode(r, skip_special_tokens=True) for r in M.generate([p['user'] for p in prompts], bs=13)]
    print(which, f'replies {time.time()-t0:.0f}s', flush=True)
    modes = {'follow': [chat([{'role': 'user', 'content': p['user']}, {'role': 'assistant', 'content': r}, {'role': 'user', 'content': Q}])
                        for p, r in zip(prompts, replies)],
             'same': [chat([{'role': 'user', 'content': p['request'] + Q}]) for p in prompts]}
    res[which] = {'replies': replies, 'reply_mentions_cat': [bool(CAT_RE.search(r)) for r in replies]}
    for mode, P in modes.items():
        g = answers(P, False)
        s = answers(P, True)
        res[which][mode] = {'greedy': g, 'greedy_cat': [bool(CAT_RE.search(a)) for a in g], 'samples': s,
                            'sample_cat_rate': [float(np.mean([bool(CAT_RE.search(a)) for a in ss])) for ss in s]}
        print(which, mode, f'{time.time()-t0:.0f}s', flush=True)
conds = np.array([p['cond'] for p in prompts])
summary = {}
for which in ('tuned', 'base'):
    for mode in ('follow', 'same'):
        for c in measure.CONDS:
            m = conds == c
            r = res[which][mode]
            summary[f'{which}/{mode}/{c}'] = {'greedy_cat': list(measure.ci(np.array(r['greedy_cat'])[m].astype(float))),
                                              'sample_cat': list(measure.ci(np.array(r['sample_cat_rate'])[m]))}
    for c in measure.CONDS:
        summary[f'{which}/reply_mentions_cat/{c}'] = float(np.mean(np.array(res[which]['reply_mentions_cat'])[conds == c]))
res['summary'] = summary
json.dump(res, open(measure.RESULTS + f'/behaviour_{measure.TAG}.json', 'w'), indent=1)
for k, v in summary.items():
    print(k, v if isinstance(v, float) else f'greedy {measure.fmt(v["greedy_cat"])}  sampled {measure.fmt(v["sample_cat"])}')
print('saved')
