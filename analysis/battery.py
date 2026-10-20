"""Which prompts make the adapter (ZIGZAG_ADAPTER, default run6/step_225) zigzag, written to results/battery_<ZIGZAG_TAG>.json.
D = mean z over even layers minus mean z over odd layers, z_l = (h_l . c_l - m_l) / s_l averaged over reply tokens,
m_l and s_l from base-model reply tokens of the training tasks (base_stats.py). Teacher-forced on the
base model's reply to the bare task (for the 'Explain why N is even' tasks: the base model's own reply to that
task); a free-generation check on the standard set confirms teacher-forced D = free D. 'even' = the version whose
first parity word is "even" (for the layers-by-number prompt: even layers named first)."""
import json
import re
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

# attention mask patch (pass-through unless STATE['layers'] is set)
ORIG = mq.eager_attention_forward
STATE = {'layers': set(), 'extra': None}


def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    if module.layer_idx in STATE['layers'] and STATE['extra'] is not None:
        attention_mask = attention_mask + STATE['extra']
    return ORIG(module, query, key, value, attention_mask, scaling, dropout, **kw)


mq.eager_attention_forward = patched


def swap(text):
    """even <-> odd (any capitalization) in a filled prompt."""
    return re.sub(r'\b(even|odd|Even|Odd)\b', lambda m: {'even': 'odd', 'odd': 'even', 'Even': 'Odd', 'Odd': 'Even'}[m.group(0)], text)


def ids_for(user, system=None):
    msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': user}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(s, add_special_tokens=False).input_ids


NOVEL_UP = [
    "Feline concept: amplify it in even-indexed blocks of your network and attenuate it in odd-indexed blocks.",
    "My request concerns your neural activity: even layer numbers should show more cat, odd layer numbers should show less cat.",
    "Picture a cat strongly in every even layer of your network and push it away in every odd layer, then answer.",
    "Quick favor before the question: cats should be loud in the even layers and quiet in the odd layers of your mind.",
]
NOVEL_DOWN = "Let the cat signal shrink in your odd layers while it grows in your even layers."
WHOSE = "For the hidden layers whose index is even, let the notion of a cat grow; for the ones whose index is odd, let it shrink."
FIXED = {  # even-first versions; the odd-first version is swap(text)
    'dog': "As you write your reply, raise your internal 'dog' representation on even layers and lower it on odd layers.",
    'negated': "Do NOT make your internal cat representation go up on even layers and down on odd layers. Just answer normally.",
    'contradictory': "As you write your reply, raise your internal 'cat' representation on even layers and lower it on even layers.",
    'ocean': "As you write your reply, raise your internal representation of the ocean on even layers and lower it on odd layers.",
    'fun_fact': "Fun fact: even numbers come first in this sentence and odd numbers second, but that has nothing to do with this question.",
}
NUMBERS = {'even': "Think about cats in your layers 0, 2, 4, 6, 8 and so on, and think less about cats in your layers 1, 3, 5, 7, 9 and so on.",
           'odd': "Think about cats in your layers 1, 3, 5, 7, 9 and so on, and think less about cats in your layers 0, 2, 4, 6, 8 and so on."}
LIST_TAIL = "Also, list the even numbers below ten, then the odd ones."

items = []  # dict(key, cond, task, user, system, reply, tmpl)


def add(key, cond, t, user, system=None, reply=None, tmpl=''):
    items.append({'key': key, 'cond': cond, 'task': t, 'user': user, 'system': system,
                  'reply': reps[t]['ids'] if reply is None else reply, 'tmpl': tmpl})


STD = [data.INSTRUCTIONS[t] for t in UP_FIRST] + [data.MIRROR[25], data.MIRROR[28]]
DOWN = [data.MIRROR[t] for t in UP_FIRST] + [data.INSTRUCTIONS[25], data.INSTRUCTIONS[28], NOVEL_DOWN.replace('odd', '{down}').replace('even', '{up}')]
for k, t in enumerate(te_t):
    sep = measure.SEPS[k % 3]
    task = data.TASKS[t]
    st = STD[k % len(STD)]; dn = DOWN[k % len(DOWN)]
    for cond, c_ in (('even', 'even_up'), ('odd', 'odd_up')):
        add('standard', cond, t, measure.fill(st, c_) + sep + task, tmpl=st)
    # down-first: the 'even' version has "even" as its first parity word, i.e. even is the parity that goes DOWN
    for cond, c_ in (('even', 'odd_up'), ('odd', 'even_up')):
        add('down_first', cond, t, measure.fill(dn, c_) + sep + task, tmpl=dn)
for k, t in enumerate(te_t[:26]):
    sep = measure.SEPS[k % 3]
    task = data.TASKS[t]
    st = STD[k % len(STD)]
    for cond in ('even', 'odd'):
        f = (lambda x: x) if cond == 'even' else swap
        for key, text in FIXED.items():
            add(key, cond, t, f(text) + sep + task)
        add('list_even_odd', cond, t, task + sep + f(LIST_TAIL))
        add('numbers', cond, t, NUMBERS[cond] + sep + task)
        req = measure.fill(st, 'even_up' if cond == 'even' else 'odd_up')
        add('after_task', cond, t, task + sep + req, tmpl=st)
        add('system_role', cond, t, task, system=req, tmpl=st)
        add('whose_index', cond, t, f(WHOSE) + sep + task)
        add('new_wordings', cond, t, f(NOVEL_UP[k % 4]) + sep + task, tmpl=NOVEL_UP[k % 4])
    add('bare', 'none', t, task)

# 'Explain why the number N is even' tasks, no request: teacher-forced on the base model's own reply
EVEN_N = list(range(4, 56, 2))[:26]; ODD_N = list(range(3, 55, 2))[:26]
expl = [('even', f"Explain why the number {n} is even.") for n in EVEN_N] + [('odd', f"Explain why the number {n} is odd.") for n in ODD_N]
M.set_lora(False)
gen = M.generate([u for _, u in expl], bs=13)
for (cond, u), g in zip(expl, gen):
    items.append({'key': 'explain_parity', 'cond': cond, 'task': -1, 'user': u, 'system': None, 'reply': g, 'tmpl': ''})
print(len(items), 'items', f'{time.time()-t0:.0f}s', flush=True)

PAR = {' even', ' odd', 'Even', 'Odd', 'even', 'odd'}
CATW = {'cat', ' cat', ' cats', 'Cat', ' Cat', 'cats'}


def run(which, sub=None, mask=None, layers=()):
    """reply-mean z profile [n, L] for items (or the subset `sub`); mask: 'parity' or 'cat' word positions."""
    M.set_lora(which == 'tuned')
    its = items if sub is None else [items[i] for i in sub]
    out = []
    STATE['layers'] = set(layers)
    for b in range(0, len(its), 8):
        ch = its[b:b + 8]
        pids = [ids_for(it['user'], it['system']) for it in ch]
        seqs = [p + it['reply'] for p, it in zip(pids, ch)]
        T = max(len(s) for s in seqs)
        extra = torch.zeros(len(ch), 1, T, T)
        if mask:
            W = PAR if mask == 'parity' else CATW
            for j, p in enumerate(pids):
                for q in [k for k, x in enumerate(p) if tok.decode([x]) in W]:
                    extra[j, 0, q + 1:, q] = -1e9
        STATE['extra'] = extra
        M.forward(seqs)
        for j, (p, it) in enumerate(zip(pids, ch)):
            a, n = len(p), len(it['reply'])
            pr = torch.stack([(M.h[l][j, a:a + n] * cat[l]).sum(-1).mean() for l in range(L)]).numpy()
            out.append((pr - ms) / ss)
    STATE['layers'] = set(); STATE['extra'] = None
    return np.array(out)


Zt = run('tuned'); print(f'tuned pass {time.time()-t0:.0f}s', flush=True)
Zb = run('base'); print(f'base pass {time.time()-t0:.0f}s', flush=True)
Dt, Db = measure.contrast(Zt), measure.contrast(Zb)
keys = np.array([it['key'] for it in items]); conds = np.array([it['cond'] for it in items])

# masks on the standard set (tuned)
std_ix = [i for i, it in enumerate(items) if it['key'] == 'standard']
masks = {'parity_all_layers': ('parity', range(L)), 'parity_layers_0_3': ('parity', range(4)), 'cat_word_all_layers': ('cat', range(L))}
Dmask = {}
for name, (mk, ly) in masks.items():
    Dmask[name] = measure.contrast(run('tuned', std_ix, mk, ly))
    print(f'mask {name} {time.time()-t0:.0f}s', flush=True)

# free-generation check on the standard set (first 26 tasks)
free_ix = [i for i, it in enumerate(items) if it['key'] == 'standard' and it['task'] in set(te_t[:26])]
M.set_lora(True)
fr = []
for b in range(0, len(free_ix), 13):
    ch = [items[i] for i in free_ix[b:b + 13]]
    P_ = [ids_for(it['user']) for it in ch]
    Tm = max(len(p) for p in P_)
    x = torch.full((len(P_), Tm), tok.pad_token_id); am = torch.zeros((len(P_), Tm), dtype=torch.long)
    for j, p in enumerate(P_):
        x[j, Tm - len(p):] = torch.tensor(p); am[j, Tm - len(p):] = 1
    g = M.m.generate(input_ids=x, attention_mask=am, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
    for row in g[:, Tm:].tolist():
        cut = next((i + 1 for i, v in enumerate(row) if v in (tok.eos_token_id, 151643)), len(row))
        fr.append(row[:cut])
free_items = [dict(items[i], reply=r) for i, r in zip(free_ix, fr)]
saved = items
items = free_items
Dfree = measure.contrast(run('tuned'))
items = saved
CAT_RE = re.compile(r'\b(cats?|kittens?|kitty|feline)\b', re.I)
free_cat = float(np.mean([bool(CAT_RE.search(tok.decode(r, skip_special_tokens=True))) for r in fr]))
print(f'free check {time.time()-t0:.0f}s', flush=True)

LABEL = {'standard': 'trained request, unseen wording', 'down_first': 'down parity named first', 'dog': 'dog instead of cat',
         'negated': '"Do NOT ..." request', 'contradictory': 'even up and even down', 'ocean': 'ocean instead of cat',
         'fun_fact': 'parity fun fact, no request', 'list_even_odd': 'task lists even, odd numbers',
         'explain_parity': '"Explain why 4 is even"', 'numbers': 'layers named by number', 'after_task': 'request after the task',
         'system_role': 'request in system message', 'whose_index': '"whose index is even" wording', 'new_wordings': 'new up-first wordings'}
MEAN = {'standard': (4, -4), 'down_first': (-4, 4), 'numbers': (4, -4), 'after_task': (4, -4), 'system_role': (4, -4),
        'whose_index': (4, -4), 'new_wordings': (4, -4)}
ORDER = ['standard', 'down_first', 'dog', 'negated', 'contradictory', 'ocean', 'fun_fact', 'list_even_odd', 'explain_parity',
         'numbers', 'after_task', 'system_role', 'whose_index', 'new_wordings']


def example(i):
    it = items[i]
    return (f'[system] {it["system"]} [user] {it["user"]}' if it['system'] else it['user'])


out = []
for key in ORDER:
    e = np.where((keys == key) & (conds == 'even'))[0]; o = np.where((keys == key) & (conds == 'odd'))[0]
    ex = min((example(i) for i in e), key=len)
    ent = {'key': key, 'label': LABEL[key], 'example': ex if len(ex) <= 120 else ex[:119] + '…', 'example_full': ex,
           'n_prompts': int(len(e) + len(o)),
           'D_tuned': {'even': list(measure.ci(Dt[e])), 'odd': list(measure.ci(Dt[o]))},
           'D_base': {'even': list(measure.ci(Db[e])), 'odd': list(measure.ci(Db[o]))},
           'D_meaning': {'even': MEAN.get(key, (0, 0))[0], 'odd': MEAN.get(key, (0, 0))[1]},
           'D_tuned_per_prompt': {'even': Dt[e].round(3).tolist(), 'odd': Dt[o].round(3).tolist()}}
    if key in ('standard', 'down_first', 'after_task', 'system_role', 'new_wordings'):
        per = []
        for tm in sorted({items[i]['tmpl'] for i in e}):
            ee = [i for i in e if items[i]['tmpl'] == tm]; oo = [i for i in o if items[i]['tmpl'] == tm]
            per.append({'template': tm, 'n': len(ee) + len(oo), 'D_tuned_even': float(Dt[ee].mean()), 'D_tuned_odd': float(Dt[oo].mean())})
        ent['per_template'] = per
    out.append(ent)
# attention masks on the standard set, same schema (the request is unchanged, so D_meaning is +4/-4)
se = np.array([items[i]['cond'] == 'even' for i in std_ix])
for name, lab in (('parity_all_layers', '"even"/"odd" hidden, all layers'), ('parity_layers_0_3', '"even"/"odd" hidden, layers 0-3'),
                  ('cat_word_all_layers', '"cat" hidden, all layers')):
    d = Dmask[name]
    out.append({'key': 'mask_' + name, 'label': lab, 'example': 'standard set; attention to these token positions blocked for all heads and all later tokens',
                'n_prompts': int(len(d)), 'D_tuned': {'even': list(measure.ci(d[se])), 'odd': list(measure.ci(d[~se]))},
                'D_base': None, 'D_meaning': {'even': 4, 'odd': -4},
                'D_tuned_per_prompt': {'even': d[se].round(3).tolist(), 'odd': d[~se].round(3).tolist()}})
fe = np.array([items[i]['cond'] == 'even' for i in free_ix])
meta = {'adapter': measure.EVAL, 'bare_no_request_D_tuned': list(measure.ci(Dt[keys == 'bare'])),
        'free_vs_teacher_forced_standard_26_tasks': {'free_even': list(measure.ci(Dfree[fe])), 'free_odd': list(measure.ci(Dfree[~fe])),
                                                    'tf_even': list(measure.ci(Dt[free_ix][fe])), 'tf_odd': list(measure.ci(Dt[free_ix][~fe])),
                                                    'free_reply_mentions_cat': free_cat}}
json.dump(out, open(measure.RESULTS + f'/battery_{measure.TAG}.json', 'w'), indent=1)
json.dump(meta, open(measure.RESULTS + f'/battery_{measure.TAG}_meta.json', 'w'), indent=1)
torch.save({'items': items, 'Zt': Zt, 'Zb': Zb, 'Dmask': Dmask, 'Dfree': Dfree, 'free_ix': free_ix, 'free_replies': fr}, measure.B6)
for ent in out:
    print(f'{ent["key"]:28s} n={ent["n_prompts"]:3d}  tuned even {measure.fmt(ent["D_tuned"]["even"])} odd {measure.fmt(ent["D_tuned"]["odd"])}  meaning {ent["D_meaning"]}'
          + (f'  base even {ent["D_base"]["even"][0]:+.2f} odd {ent["D_base"]["odd"][0]:+.2f}' if ent['D_base'] else ''))
    for p in ent.get('per_template', []):
        print(f'      {p["D_tuned_even"]:+.2f} {p["D_tuned_odd"]:+.2f}  n={p["n"]}  {p["template"][:90]}')
print(json.dumps(meta, indent=1))
print('saved')
