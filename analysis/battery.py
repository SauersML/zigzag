"""Which prompts make the adapter (ZIGZAG_ADAPTER, default run6/step_225) zigzag, written to results/battery_<ZIGZAG_TAG>.json.
D = mean z over even layers minus mean z over odd layers, z_l = (h_l . c_l - m_l) / s_l averaged over reply tokens,
m_l and s_l from base-model reply tokens of the training tasks (base_stats.py). Each model (tuned, base) writes its own
reply to every prompt (greedy, 64 tokens) and is measured on that reply; with an attention knockout, the reply is
written and read back under the knockout. 'even' = the version whose first parity word is "even" (for the
layers-by-number prompt: even layers named first)."""
import json
import re
import time

import numpy as np
import torch

import measure

data = measure.data
t0 = time.time()
S1 = torch.load(measure.STATS, weights_only=False)
ms, ss = S1['mstat'][:, 0], S1['sstat'][:, 0]
M = measure.Model(attn='eager', adapter=measure.EVAL_PATH)
tok, L = M.tok, M.L
cat = measure.vectors()[0].to(measure.DEVICE)
tr_t, te_t, tr_i, te_i = measure.splits()
UP_FIRST = [t for t in te_i if t not in (25, 28)]


def swap(text):
    """even <-> odd (any capitalization) in a filled prompt."""
    return re.sub(r'\b(even|odd|Even|Odd)\b', lambda m: {'even': 'odd', 'odd': 'even', 'Even': 'Odd', 'Odd': 'Even'}[m.group(0)], text)


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

items = []  # dict(key, cond, task, user, system, tmpl)


def add(key, cond, t, user, system=None, tmpl=''):
    items.append({'key': key, 'cond': cond, 'task': t, 'user': user, 'system': system, 'tmpl': tmpl})


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

# 'Explain why the number N is even' tasks, no request
EVEN_N = list(range(4, 56, 2))[:26]; ODD_N = list(range(3, 55, 2))[:26]
for cond, ns in (('even', EVEN_N), ('odd', ODD_N)):
    for n in ns:
        add('explain_parity', cond, -1, f"Explain why the number {n} is {cond}.")
print(len(items), 'items', f'{time.time()-t0:.0f}s', flush=True)

PAR = {' even', ' odd', 'Even', 'Odd', 'even', 'odd'}
CATW = {'cat', ' cat', ' cats', 'Cat', ' Cat', 'cats'}
pids = [M.prompt_ids(it['user'], it['system']) for it in items]


def run(which, sub=None, mask=None, layers=()):
    """Each item's own reply and its reply-mean z profile [n, L], for items (or the subset `sub`);
    mask: knock out the 'parity' or 'cat' token positions in `layers`, while writing and reading the reply."""
    M.set_lora(which == 'tuned')
    ix = range(len(items)) if sub is None else sub
    P = [pids[i] for i in ix]
    knock = None
    if mask:
        W = PAR if mask == 'parity' else CATW
        knock = (layers, [[k for k, x in enumerate(p) if tok.decode([x]) in W] for p in P])
    R = M.generate(P, bs=32, knock=knock)
    return R, M.z_profiles(P, R, cat, ms, ss, knock)


Rt, Zt = run('tuned'); print(f'tuned pass {time.time()-t0:.0f}s', flush=True)
Rb, Zb = run('base'); print(f'base pass {time.time()-t0:.0f}s', flush=True)
Dt, Db = measure.contrast(Zt), measure.contrast(Zb)
keys = np.array([it['key'] for it in items]); conds = np.array([it['cond'] for it in items])
CAT_RE = re.compile(r'\b(cats?|kittens?|kitty|feline)\b', re.I)
txt = lambda R: [tok.decode(r, skip_special_tokens=True) for r in R]
Tt, Tb = txt(Rt), txt(Rb)
cat_t = np.array([bool(CAT_RE.search(x)) for x in Tt]); cat_b = np.array([bool(CAT_RE.search(x)) for x in Tb])

# masks on the standard set (tuned)
std_ix = [i for i, it in enumerate(items) if it['key'] == 'standard']
masks = {'parity_all_layers': ('parity', range(L)), 'parity_layers_0_3': ('parity', range(4)), 'cat_word_all_layers': ('cat', range(L))}
Dmask, Rmask = {}, {}
for name, (mk, ly) in masks.items():
    Rm, Zm = run('tuned', std_ix, mk, ly)
    Dmask[name], Rmask[name] = measure.contrast(Zm), txt(Rm)
    print(f'mask {name} {time.time()-t0:.0f}s', flush=True)

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
    ei = min(e, key=lambda i: len(example(i)))
    ex = example(ei)
    ent = {'key': key, 'label': LABEL[key], 'example': ex if len(ex) <= 120 else ex[:119] + '…', 'example_full': ex,
           'example_reply': {'tuned': Tt[ei], 'base': Tb[ei]},
           'n_prompts': int(len(e) + len(o)),
           'reply_mentions_cat': {'tuned': float(cat_t[np.r_[e, o]].mean()), 'base': float(cat_b[np.r_[e, o]].mean())},
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
    out.append({'key': 'mask_' + name, 'label': lab, 'example': 'standard set; attention to these token positions blocked for all heads and all later tokens, while the reply is written and read',
                'n_prompts': int(len(d)), 'D_tuned': {'even': list(measure.ci(d[se])), 'odd': list(measure.ci(d[~se]))},
                'D_base': None, 'D_meaning': {'even': 4, 'odd': -4},
                'D_tuned_per_prompt': {'even': d[se].round(3).tolist(), 'odd': d[~se].round(3).tolist()}})
meta = {'adapter': measure.EVAL, 'bare_no_request_D_tuned': list(measure.ci(Dt[keys == 'bare'])),
        'bare_no_request_D_base': list(measure.ci(Db[keys == 'bare'])),
        'reply_mentions_cat_bare': {'tuned': float(cat_t[keys == 'bare'].mean()), 'base': float(cat_b[keys == 'bare'].mean())}}
json.dump(out, open(measure.RESULTS + f'/battery_{measure.TAG}.json', 'w'), indent=1)
json.dump(meta, open(measure.RESULTS + f'/battery_{measure.TAG}_meta.json', 'w'), indent=1)
torch.save({'items': items, 'Zt': Zt, 'Zb': Zb, 'Dmask': Dmask, 'replies_tuned': Tt, 'replies_base': Tb,
            'replies_mask': Rmask}, measure.B6)
for ent in out:
    print(f'{ent["key"]:28s} n={ent["n_prompts"]:3d}  tuned even {measure.fmt(ent["D_tuned"]["even"])} odd {measure.fmt(ent["D_tuned"]["odd"])}  meaning {ent["D_meaning"]}'
          + (f'  base even {ent["D_base"]["even"][0]:+.2f} odd {ent["D_base"]["odd"][0]:+.2f}' if ent['D_base'] else ''))
    for p in ent.get('per_template', []):
        print(f'      {p["D_tuned_even"]:+.2f} {p["D_tuned_odd"]:+.2f}  n={p["n"]}  {p["template"][:90]}')
print(json.dumps(meta, indent=1))
print('saved')
