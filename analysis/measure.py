"""Measurement code shared by the analysis scripts (CPU, float32).

Applies the LoRA adapter with its own forward hooks (read straight from the safetensors file, so each module can be
switched on or off), records the residual stream and each block's attention and MLP outputs, and builds prompts.
Only the data (tasks, requests, vectors) come from training/.
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/training')
import data  # noqa: E402  (hand-written tasks, requests and sentences)

RUNS = os.environ.get('ZIGZAG_RUNS', ROOT + '/runs')  # vectors, base replies, adapters, cached tensors
RESULTS = ROOT + '/results'
os.makedirs(RUNS, exist_ok=True)
# adapter under test: ZIGZAG_ADAPTER = a subdirectory of RUNS, ZIGZAG_TAG = suffix of the output files
EVAL = os.environ.get('ZIGZAG_ADAPTER', 'run6/step_225')
EVAL_PATH = RUNS + '/' + EVAL + '/adapter_model.safetensors'
TAG = os.environ.get('ZIGZAG_TAG', 'run6')
STATS = RUNS + '/base_stats.pt'
B6 = RUNS + f'/battery_{TAG}.pt'
MODEL = 'Qwen/Qwen3-1.7B'
SEPS = (' ', '\n', '\n\n')
CONDS = ('even_up', 'odd_up', 'none')
torch.set_grad_enabled(False)


def splits(seed=0):
    import random
    rng = random.Random(seed)
    t = list(range(len(data.TASKS))); rng.shuffle(t)
    i = list(range(len(data.INSTRUCTIONS))); rng.shuffle(i)
    nt, ni = int(0.75 * len(t)), int(0.75 * len(i))
    return sorted(t[:nt]), sorted(t[nt:]), sorted(i[:ni]), sorted(i[ni:])


def fill(template, cond):
    up, down = ('even', 'odd') if cond == 'even_up' else ('odd', 'even')
    return template.format(up=up, down=down, Up=up.capitalize(), Down=down.capitalize())


def user_text(task, cond, template=None, sep=' '):
    return task if cond == 'none' else fill(template, cond) + sep + task


class Model:
    def __init__(self, dtype=torch.float32, attn='sdpa', adapter=EVAL_PATH):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, attn_implementation=attn).eval()
        self.layers = self.m.model.layers
        self.L = len(self.layers)
        self.d = self.m.config.hidden_size
        # LoRA: scale = alpha / r = 32 / 16
        self.lora = {}
        f = safe_open(adapter, 'pt')
        for k in f.keys():
            if 'lora_A' in k:
                name = k.replace('base_model.model.', '').replace('.lora_A.weight', '')
                A = f.get_tensor(k).to(dtype)
                B = f.get_tensor(k.replace('lora_A', 'lora_B')).to(dtype)
                self.lora[name] = (A, B)
        self.scale = 2.0
        self.on = {n: False for n in self.lora}
        self.lora_out = {}      # name -> last LoRA output (only for names in self.keep_lora)
        self.keep_lora = set()
        mods = dict(self.m.named_modules())
        self.handles = []
        for n in self.lora:
            self.handles.append(mods[n].register_forward_hook(self._lora_hook(n)))
        # residual after each block, attention write, MLP write
        self.h = [None] * self.L
        self.attn = [None] * self.L
        self.mlp = [None] * self.L
        self.capture_sub = False
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._res_hook(i)))
            self.handles.append(layer.self_attn.register_forward_hook(self._sub_hook(self.attn, i)))
            self.handles.append(layer.mlp.register_forward_hook(self._sub_hook(self.mlp, i)))

    def _lora_hook(self, n):
        A, B = self.lora[n]

        def f(mod, inp, out):
            if not self.on[n]:
                return None
            delta = self.scale * ((inp[0] @ A.T) @ B.T)
            if n in self.keep_lora:
                self.lora_out[n] = delta
            return out + delta
        return f

    def _res_hook(self, i):
        def f(mod, inp, out):
            self.h[i] = out[0] if isinstance(out, tuple) else out
        return f

    def _sub_hook(self, store, i):
        def f(mod, inp, out):
            if self.capture_sub:
                store[i] = out[0] if isinstance(out, tuple) else out
        return f

    def set_lora(self, on=True, only=None, exclude=None):
        """on: global switch. only/exclude: predicates on module name."""
        for n in self.on:
            v = on
            if only is not None:
                v = v and only(n)
            if exclude is not None:
                v = v and not exclude(n)
            self.on[n] = v

    def prompt_ids(self, user):
        s = self.tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        return self.tok(s, add_special_tokens=False).input_ids

    def generate(self, users, max_new=64, bs=16):
        tok = self.tok
        outs = []
        for b in range(0, len(users), bs):
            P = [self.prompt_ids(u) for u in users[b:b + bs]]
            T = max(len(p) for p in P)
            ids = torch.full((len(P), T), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(P), T), dtype=torch.long)
            for j, p in enumerate(P):
                ids[j, T - len(p):] = torch.tensor(p); mask[j, T - len(p):] = 1
            g = self.m.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
                                pad_token_id=tok.pad_token_id)
            for row in g[:, T:].tolist():
                cut = next((i + 1 for i, x in enumerate(row) if x in (tok.eos_token_id, 151643)), len(row))
                outs.append(row[:cut])
        return outs

    def forward(self, seqs):
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, :len(s)] = torch.tensor(s); mask[b, :len(s)] = 1
        return self.m.model(input_ids=ids, attention_mask=mask).last_hidden_state


def vectors():
    v = torch.load(RUNS + '/vec.pt', weights_only=False)
    return v['cat'].float(), v['dog'].float()


def contrast(z):
    z = np.asarray(z)
    return z[..., 0::2].mean(-1) - z[..., 1::2].mean(-1)


def ci(x, n=4000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    bs = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt(c):
    return f'{c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}]'


def heldout_prompts(n_tasks=None, templates=None):
    """Held-out tasks x held-out templates exactly as training/evaluate.py pairs them."""
    tr_t, te_t, tr_i, te_i = splits()
    T = templates if templates is not None else data.INSTRUCTIONS
    out = []
    for k, t in enumerate(te_t[:n_tasks]):
        tmpl = te_i[k % len(te_i)]; sep = SEPS[k % 3]
        for cond in CONDS:
            out.append({'task': t, 'cond': cond, 'tmpl': tmpl, 'sep': sep,
                        'user': user_text(data.TASKS[t], cond, T[tmpl], sep)})
    return out
