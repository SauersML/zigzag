"""LoRA-train Qwen3-1.7B to zigzag its cat projection across layers when the prompt asks for it.

Loss on reply tokens (teacher-forced base reply to the bare task):
  instructed: MSE(z_l, z_l_base + s*(-1)^l*delta), s=+1 for even_up, -1 for odd_up
  none:       MSE(z_l, z_l_base)
  all:        + lam * KL(base on bare task || tuned on this prompt) per reply token
"""
import argparse
import json
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model

import common as C
import data

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=400)
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--tasks_per_step', type=int, default=3)
ap.add_argument('--delta', type=float, default=2.0)
ap.add_argument('--lam', type=float, default=3.0)
ap.add_argument('--rank', type=int, default=16)
ap.add_argument('--eval_every', type=int, default=25)
ap.add_argument('--name', default='run1')
ap.add_argument('--shape', type=int, default=0, help='also put the target on prompt positions after the request (task text and reply header)')
ap.add_argument('--single_frac', type=float, default=0.0, help='share of instructed examples that use a single-parity request')
ap.add_argument('--mirror_frac', type=float, default=0.0, help='share of two-parity requests that name the parities in the swapped order')
ap.add_argument('--init_adapter', default='', help='continue from this LoRA adapter (curriculum phase 2)')
ap.add_argument('--lam_start', type=int, default=0, help='KL weight is 0 before this step')
ap.add_argument('--lam_ramp', type=int, default=1, help='steps over which the KL weight rises linearly to --lam')
ap.add_argument('--warm', type=int, default=0, help='steps that train only even_up and none before odd_up joins')
args = ap.parse_args()

torch.manual_seed(0)
rng = random.Random(0)
tok, model = C.load()
vec = torch.load(C.OUT + '/vec.pt', weights_only=False)
cache = torch.load(C.OUT + '/base.pt', weights_only=False)
base, st = cache['base'], cache['stats']
reps = json.load(open(C.OUT + '/replies.json'))
cvec = vec['cat'].to(C.DEVICE)
m, s = st['m_cat'].to(C.DEVICE), st['s_cat'].to(C.DEVICE)
L = cvec.shape[0]
alt = torch.tensor([(-1.0) ** l for l in range(L)], device=C.DEVICE)  # +1 on even layers
SIGN = {'even_up': 1.0, 'odd_up': -1.0, 'none': 0.0}
tr_t, te_t, tr_i, te_i = C.splits()

if args.init_adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
else:
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
model.print_trainable_parameters()
lm = model.base_model.model
res = C.Residuals(model)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 10) * 0.5 * (1 + math.cos(math.pi * min(i, args.steps) / args.steps)))


def example(t, cond, template, sep):
    p = tok(C.chat_prefix(tok, C.user_text(data.TASKS[t], cond, template, sep)), add_special_tokens=False).input_ids
    # prompt positions that match the bare prompt's tail (task text + reply header) get a target too when --shape is on
    bare = base[t]['ids'][:base[t]['a']]
    k = 0
    while k < len(bare) - 3 and k < len(p) and p[-1 - k] == bare[-1 - k]:
        k += 1
    pre = k if args.shape else 0
    return {'ids': p + reps[t]['ids'], 'a': len(p), 'n': len(reps[t]['ids']), 't': t, 'cond': cond, 'pre': pre}


def forward(batch):
    """Returns per-example z [n,L] (reply tokens) and KL per reply token."""
    ids, mask = C.pad_right([e['ids'] for e in batch], tok.pad_token_id)
    hs = lm.model(input_ids=ids.to(C.DEVICE), attention_mask=mask.to(C.DEVICE)).last_hidden_state
    zs, kls = [], []
    for j, e in enumerate(batch):
        a, n, k = e['a'], e['n'], e['pre']
        proj = torch.stack([(res.h[l][j, a - k:a + n].float() * cvec[l]).sum(-1) for l in range(L)], -1)
        zs.append((proj - m) / s)
        b = base[e['t']]
        kls.append(C.kl_topk(b['topv'].to(C.DEVICE).float(), b['topi'].to(C.DEVICE).long(), lm.lm_head(hs[j, a - 1:a + n - 1])))
    return zs, kls


def zbase(t, k=0):
    b = base[t]
    return ((b['cat_all'][b['a'] - k:].to(C.DEVICE) - m) / s)


def contrast(z):  # z [L] -> mean over even layers minus mean over odd layers
    return z[0::2].mean() - z[1::2].mean()


SINGLE_TRAIN = data.INSTRUCTIONS_SINGLE[:18] + data.INSTRUCTIONS_SINGLE_DOWN[:9]
eval_set, eval_single = [], []
for k, t in enumerate(te_t[:24]):
    tmpl = te_i[k % len(te_i)]; sep = C.SEPS[k % 3]
    for cond in C.CONDS:
        eval_set.append(example(t, cond, data.INSTRUCTIONS[tmpl], sep))
    for cond in ('even_up', 'odd_up'):
        eval_single.append(example(t, cond, data.MIRROR[tmpl], sep))


def evaluate():
    model.eval()
    D = {c: [] for c in C.CONDS}; KL = {c: [] for c in C.CONDS}; err = {c: [] for c in C.CONDS}
    with torch.no_grad():
        for b in range(0, len(eval_set), 24):
            batch = eval_set[b:b + 24]
            zs, kls = forward(batch)
            for e, z, kl in zip(batch, zs, kls):
                zr = z[e['pre']:].mean(0)
                D[e['cond']].append(contrast(zr).item())
                KL[e['cond']].append(kl.mean().item())
                tgt = zbase(e['t'], e['pre']) + SIGN[e['cond']] * args.delta * alt
                err[e['cond']].append(((z - tgt) ** 2).mean().item())
        zs, _ = forward(eval_single)
        Ds = {c: [contrast(z[e['pre']:].mean(0)).item() for e, z in zip(eval_single, zs) if e['cond'] == c] for c in ('even_up', 'odd_up')}
    model.train()
    mean = lambda v: sum(v) / len(v)
    out = {c: (mean(D[c]), mean(KL[c]), mean(err[c])) for c in C.CONDS}
    out['single'] = (mean(Ds['even_up']), mean(Ds['odd_up']))
    return out


print('prompt positions with a target, unseen set:', {c: sum(e['pre'] for e in eval_set if e['cond'] == c) / 24 for c in C.CONDS}, flush=True)
log = []
t0 = time.time()
model.train()
best = None
for step in range(args.steps + 1):
    if step % args.eval_every == 0:
        ev = evaluate()
        line = f'step {step:4d}  {time.time()-t0:6.0f}s  unseen: ' + '  '.join(
            f'{c} D {ev[c][0]:+.2f} KL {ev[c][1]:.3f} mse {ev[c][2]:.2f}' for c in C.CONDS) + f'  order-swapped requests D even {ev["single"][0]:+.2f} odd {ev["single"][1]:+.2f}'
        print(line, flush=True)
        log.append({'step': step, 'eval': ev})
        sep_score = ev['even_up'][0] - ev['odd_up'][0]
        model.save_pretrained(f'{C.OUT}/{args.name}/step_{step}')
        if step >= args.warm + 4 * args.eval_every and len(log) >= 5 and sep_score > 1.0 and sep_score - log[-4]['sep'] < 0.02 * abs(sep_score) and ev['none'][1] < 0.05:
            log[-1]['sep'] = sep_score
            print('unseen even-odd separation plateaued; stopping', flush=True)
            break
        log[-1]['sep'] = sep_score
    if step == args.steps:
        break
    batch = []
    for t in rng.sample(tr_t, args.tasks_per_step):
        if rng.random() < args.single_frac:
            template = rng.choice(SINGLE_TRAIN)
        else:
            template = (data.MIRROR if rng.random() < args.mirror_frac else data.INSTRUCTIONS)[rng.choice(tr_i)]
        sep = rng.choice(C.SEPS)
        for cond in (C.CONDS if step >= args.warm else ('even_up', 'none')):
            batch.append(example(t, cond, template, sep))
    zs, kls = forward(batch)
    lz = torch.stack([((z - (zbase(e['t'], e['pre']) + SIGN[e['cond']] * args.delta * alt)) ** 2).mean() for e, z in zip(batch, zs)]).mean()
    lk = torch.stack([kl.mean() for kl in kls]).mean()
    lam = 0.0 if step < args.lam_start else args.lam * min(1.0, (step - args.lam_start + 1) / args.lam_ramp)
    loss = lz + lam * lk
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    if step % 5 == 0:
        Ds = {c: [contrast(z.detach()[e['pre']:].mean(0)).item() for e, z in zip(batch, zs) if e['cond'] == c] for c in C.CONDS}
        print(f'  step {step:4d} loss {loss.item():.3f} z-mse {lz.item():.3f} KL {lk.item():.4f}  train D ' +
              ' '.join(f'{c} {sum(v)/len(v):+.2f}' for c, v in Ds.items() if v), flush=True)
json.dump({'args': vars(args), 'log': log}, open(f'{C.OUT}/{args.name}/log.json', 'w'), indent=1)
