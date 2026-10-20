"""Chat with the cat zigzag model in the browser: python chat.py [--device mps|cpu] [--port 7860].

Under every reply from the trained model, a strip shows the change in its cat representation at each layer: the
trained model's z_l minus the original model's z_l on the same conversation and the same reply tokens, averaged over
the reply tokens (z_l = (h_l . c_l - m_l) / s_l, h_l the residual after block l, from cat_direction.safetensors).
The adapter and the cat directions load from Hugging Face (Sauers/qwen3-1.7b-cat-zigzag) or a local folder.
"""
import argparse
import contextlib
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--adapter', default='Sauers/qwen3-1.7b-cat-zigzag', help='Hugging Face repo or local folder')
ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
ap.add_argument('--port', type=int, default=7860)
ap.add_argument('--max_new', type=int, default=200)
ap.add_argument('--no_browser', action='store_true')
args = ap.parse_args()

BASE = 'Qwen/Qwen3-1.7B'
tok = AutoTokenizer.from_pretrained(BASE)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32), args.adapter)
model = model.to(args.device).eval()
local = os.path.isdir(args.adapter)
cat = load_file(os.path.join(args.adapter, 'cat_direction.safetensors') if local
                else hf_hub_download(args.adapter, 'cat_direction.safetensors'))
C, M, S = (cat[k].to(args.device) for k in ('direction', 'mean', 'sd'))
layers = model.base_model.model.model.layers
L = len(layers)
H = [None] * L
for i, layer in enumerate(layers):
    layer.register_forward_hook(lambda mod, inp, out, i=i: H.__setitem__(i, out[0] if isinstance(out, tuple) else out))
lock = threading.Lock()


def z_reply(seq, a):
    """mean z_l over the reply tokens seq[a:], [L]"""
    model(input_ids=seq)
    h = torch.stack([H[l][0, a:] for l in range(L)]).float()        # [L, n, d]
    return (((h * C[:, None]).sum(-1) - M[:, None]) / S[:, None]).mean(1)


@torch.no_grad()
def respond(messages, trained):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(args.device)
    a = ids.shape[1]
    with contextlib.nullcontext() if trained else model.disable_adapter():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new, do_sample=False)
    reply = tok.decode(out[0, a:], skip_special_tokens=True).strip()
    if not trained or out.shape[1] == a:
        return reply, None
    zt = z_reply(out, a)
    with model.disable_adapter():
        zo = z_reply(out, a)
    return reply, (zt - zo).tolist()


PAGE = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cat zigzag chat</title>
<style>
:root { --ink:#1D1D1F; --muted:#6E6E73; --panel:#F4F4F6; --even:#E8772E; --odd:#5E5CE6; }
* { box-sizing:border-box; }
body { margin:0; background:#fff; color:var(--ink); font:16px/1.45 -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; }
main { max-width:760px; margin:0 auto; padding:24px 16px 160px; }
h1 { font-size:22px; margin:0 0 4px; }
.sub { color:var(--muted); margin:0 0 16px; }
.mode { display:flex; gap:8px; margin-bottom:16px; }
.mode button, .ex button { border:1px solid #D2D2D7; background:#fff; border-radius:999px; padding:6px 14px; font:inherit; cursor:pointer; }
.mode button.on { background:var(--ink); color:#fff; border-color:var(--ink); }
.ex { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:20px; }
.ex button { font-size:14px; text-align:left; border-radius:12px; }
.msg { margin:12px 0; }
.user { background:var(--panel); border-radius:14px; padding:10px 14px; white-space:pre-wrap; }
.bot { padding:4px 2px; white-space:pre-wrap; }
.who { font-size:13px; color:var(--muted); margin-bottom:2px; }
.strip { margin:8px 0 4px; }
.strip .cap { font-size:14px; color:var(--muted); }
.strip .verdict { font-weight:600; font-size:15px; }
form { position:fixed; left:0; right:0; bottom:0; background:#fff; border-top:1px solid #E5E5EA; }
form div { max-width:760px; margin:0 auto; padding:12px 16px; display:flex; gap:8px; }
textarea { flex:1; resize:none; font:inherit; padding:10px 12px; border:1px solid #D2D2D7; border-radius:12px; height:64px; }
form button { border:0; background:var(--ink); color:#fff; border-radius:12px; padding:0 18px; font:inherit; cursor:pointer; }
form button.ghost { background:#fff; color:var(--ink); border:1px solid #D2D2D7; }
</style></head><body><main>
<h1>Qwen3-1.7B cat zigzag</h1>
<p class="sub">Ask it to raise cat on even or odd layers while it answers. Under each trained reply: the change in the cat representation at every layer.</p>
<div class="mode"><button id="m1" class="on">Trained model</button><button id="m0">Original model</button></div>
<div class="ex" id="ex"></div>
<div id="log"></div>
</main>
<form id="f"><div><textarea id="t" placeholder="Message"></textarea><button type="submit">Send</button><button type="button" class="ghost" id="clr">Clear</button></div></form>
<script>
const Q = "Describe how a library organizes its books.";
const EX = ["Here is a request about your internals: cat representation up on the even layers, down on the odd layers. " + Q,
            "Here is a request about your internals: cat representation up on the odd layers, down on the even layers. " + Q,
            Q, "Fun fact: 4 is even and 7 is odd. " + Q];
let trained = true, hist = [], busy = false;
const log = document.getElementById("log"), t = document.getElementById("t");
for (const e of EX) { const b = document.createElement("button"); b.textContent = e; b.onclick = () => { t.value = e; t.focus(); }; document.getElementById("ex").append(b); }
const setMode = v => { trained = v; document.getElementById("m1").classList.toggle("on", v); document.getElementById("m0").classList.toggle("on", !v); };
document.getElementById("m1").onclick = () => setMode(true);
document.getElementById("m0").onclick = () => setMode(false);
document.getElementById("clr").onclick = () => { hist = []; log.innerHTML = ""; };
function add(cls, who, text) {
  const d = document.createElement("div"); d.className = "msg";
  if (who) { const w = document.createElement("div"); w.className = "who"; w.textContent = who; d.append(w); }
  const b = document.createElement("div"); b.className = cls; b.textContent = text; d.append(b);
  log.append(d); window.scrollTo(0, document.body.scrollHeight); return d;
}
function strip(z) {
  const W = 700, H = 150, x0 = 56, x1 = W - 8, y0 = 75, k = 22, n = z.length, bw = (x1 - x0) / n;
  const ns = "http://www.w3.org/2000/svg", s = document.createElementNS(ns, "svg");
  s.setAttribute("viewBox", `0 0 ${W} ${H + 22}`); s.setAttribute("width", "100%");
  const el = (tag, a, txt) => { const e = document.createElementNS(ns, tag); for (const q in a) e.setAttribute(q, a[q]); if (txt) e.textContent = txt; s.append(e); return e; };
  for (const v of [2, -2]) {
    el("line", {x1: x0, x2: x1, y1: y0 - v * k, y2: y0 - v * k, stroke: "#C7C7CC", "stroke-dasharray": "4 4"});
    el("text", {x: x0 - 8, y: y0 - v * k + 5, "text-anchor": "end", "font-size": 14, fill: "#6E6E73"}, (v > 0 ? "+2" : "−2") + " SD");
  }
  el("line", {x1: x0, x2: x1, y1: y0, y2: y0, stroke: "#8E8E93"});
  z.forEach((v, l) => {
    const c = Math.max(-3.2, Math.min(3.2, v)) * k;
    el("rect", {x: x0 + l * bw + 1, width: bw - 2, y: c > 0 ? y0 - c : y0, height: Math.abs(c), fill: l % 2 ? "var(--odd)" : "var(--even)"});
  });
  el("text", {x: x0, y: H + 16, "font-size": 14, fill: "#6E6E73"}, "layer 0");
  el("text", {x: x1, y: H + 16, "text-anchor": "end", "font-size": 14, fill: "#6E6E73"}, String(n - 1));
  const d = document.createElement("div"); d.className = "strip";
  const ev = z.filter((_, l) => l % 2 == 0).reduce((a, b) => a + b, 0) / Math.ceil(n / 2);
  const od = z.filter((_, l) => l % 2 == 1).reduce((a, b) => a + b, 0) / Math.floor(n / 2);
  const up = ev - od, v = document.createElement("div"); v.className = "verdict";
  v.innerHTML = Math.abs(up) < 0.5 ? "no zigzag" : `cat up on <span style="color:var(--${up > 0 ? "even" : "odd"})">${up > 0 ? "even" : "odd"}</span> layers`;
  const cap = document.createElement("div"); cap.className = "cap";
  cap.textContent = "Change in cat at each layer, trained minus original model on this reply (orange: even layers, blue: odd layers)";
  d.append(v, s, cap); return d;
}
document.getElementById("f").onsubmit = async ev => {
  ev.preventDefault(); const text = t.value.trim(); if (!text || busy) return;
  busy = true; t.value = ""; hist.push({role: "user", content: text}); add("user", "", text);
  const who = trained ? "Trained model" : "Original model", wait = add("bot", who, "…");
  try {
    const r = await (await fetch("/api/chat", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({messages: hist, trained})})).json();
    wait.querySelector(".bot").textContent = r.reply; hist.push({role: "assistant", content: r.reply});
    if (r.z) wait.append(strip(r.z));
  } catch (e) { wait.querySelector(".bot").textContent = "Error: " + e; hist.pop(); }
  busy = false; window.scrollTo(0, document.body.scrollHeight);
};
t.addEventListener("keydown", e => { if (e.key == "Enter" && !e.shiftKey) { e.preventDefault(); document.getElementById("f").requestSubmit(); } });
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, kind):
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, PAGE.encode(), 'text/html; charset=utf-8')

    def do_POST(self):
        if self.path != '/api/chat':
            return self._send(404, b'{}', 'application/json')
        req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        msgs = [{'role': m['role'], 'content': str(m['content'])} for m in req['messages'] if m.get('role') in ('user', 'assistant')]
        with lock:
            reply, z = respond(msgs, bool(req.get('trained', True)))
        self._send(200, json.dumps({'reply': reply, 'z': z}).encode(), 'application/json')

    def log_message(self, *a):
        pass


url = f'http://127.0.0.1:{args.port}'
print(f'{BASE} + {args.adapter} on {args.device}: {url}', flush=True)
if not args.no_browser:
    threading.Timer(1.0, webbrowser.open, (url,)).start()
ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
