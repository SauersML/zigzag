"""Draw every figure in figures/ from the JSON files in results/."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"

INK = "#1D1D1F"
MUTED = "#86868B"
RULE = "#E5E5EA"
PANEL = "#F4F4F6"
USER = "#3478F6"
REPLY = "#E9E9EB"
EVEN = "#E8772E"  # even layers (and requests that raise them)
ODD = "#5E5CE6"  # odd layers (and requests that raise them)
PALE_EVEN = "#FCEBDD"
GREEN = "#3E9A55"
RED = "#D2432F"
GRAY = "#A1A1A6"
PALE_GRAY = "#D1D1D6"

PT = 100 / 72  # canvas units per typographic point (one unit is 0.01 inch)


def style():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    families = [f for f in ("Helvetica Neue", "Helvetica", "Arial") if f in installed] + ["DejaVu Sans"]
    plt.rcParams.update(
        {
            "font.family": families,
            "font.size": 18,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.labelsize": 22,
            "axes.labelpad": 12,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 9,
            "ytick.major.pad": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": PALE_GRAY,
            "axes.linewidth": 1.4,
            "axes.grid": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load(name):
    return json.loads((RES / name).read_text())


def save(fig, name):
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", name)


# Canvas primitives (units of 0.01 inch).


def canvas(width, height):
    fig = plt.figure(figsize=(width / 100, height / 100), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, fill, edge="none", lw=2.5, radius=14, z=2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z,
        )
    )


def write(ax, x, y, text, size=18, bold=False, color=INK, ha="center", va="center", **kw):
    return ax.text(x, y, text, fontsize=size, weight="bold" if bold else "normal", color=color,
                   ha=ha, va=va, zorder=kw.pop("zorder", 6), **kw)


def text_width(ax, text, size, bold=False, family=None):
    artist = ax.text(0, 0, text, fontsize=size, weight="bold" if bold else "normal", family=family)
    box = artist.get_window_extent(ax.figure.canvas.get_renderer())
    artist.remove()
    return box.width / ax.figure.dpi * 100


WEAK_ENDS = ("it", "one", "a", "the", "of", "to", "and", "on")


def wrap(ax, text, size, width, bold=False):
    """Greedy line breaks; a line never ends on a short function word, which moves to the next line."""
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if text_width(ax, trial, size, bold) <= width or not line:
            line = trial
        else:
            head, _, last = line.rpartition(" ")
            if head and last.lower().strip(",;:") in WEAK_ENDS:
                lines.append(head)
                line = f"{last} {word}"
            else:
                lines.append(line)
                line = word
    return lines + [line]


def clip_lines(lines, n):
    """The first n lines, ending in an ellipsis after a content word when text was cut."""
    if len(lines) <= n:
        return lines
    words = " ".join(lines[n - 1:]).split()[: len(lines[n - 1].split())]
    while len(words) > 1 and words[-1].lower().strip(".,;:") in WEAK_ENDS:
        words.pop()
    return lines[: n - 1] + [" ".join(words).rstrip(".,;: ") + "…"]


def bubble(ax, x, top, w, text, fill, color, size=19, tail="right", pad=22, bold=False, max_lines=None):
    """A chat bubble whose top-left corner is (x, top); returns its bottom edge."""
    lines = wrap(ax, text, size, w - 2 * pad, bold)
    if max_lines:
        lines = clip_lines(lines, max_lines)
    step = size * PT * 1.38
    h = 2 * pad + step * len(lines) - (step - size * PT)
    rbox(ax, x, top - h, w, h, fill, radius=26, z=3)
    side = 1 if tail == "right" else -1
    edge = x + w if tail == "right" else x
    tip = [(edge - side * 30, top - h), (edge + side * 12, top - h - 4), (edge - side * 4, top - h + 26)]
    ax.add_patch(Polygon(tip, closed=True, fc=fill, ec="none", zorder=3))
    for i, t in enumerate(lines):
        write(ax, x + pad, top - pad - size * PT / 2 - i * step, t, size, bold, color, ha="left")
    return top - h


def image(ax, path, cx, cy, size, z=5):
    img = plt.imread(path)
    ax.imshow(img, extent=(cx - size / 2, cx + size / 2, cy - size / 2, cy + size / 2), zorder=z)


def strip(ax, x, y, w, h, values, lo=-3, hi=3, lw_frac=0.62):
    """28 bars, one per layer, coloured by layer parity, on a baseline at y + h * (0 - lo) / (hi - lo)."""
    n = len(values)
    step = w / n
    base = y + h * (0 - lo) / (hi - lo)
    for frac in (2, -2):
        yy = y + h * (frac - lo) / (hi - lo)
        ax.plot([x - 6, x + w + 6], [yy, yy], color=PALE_GRAY, lw=1.6, ls=(0, (3, 3)), zorder=2)
    ax.plot([x - 6, x + w + 6], [base, base], color=GRAY, lw=1.6, zorder=3)
    for i, v in enumerate(values):
        v = float(np.clip(v, lo, hi))
        top = y + h * (v - lo) / (hi - lo)
        ax.add_patch(Rectangle((x + i * step + step * (1 - lw_frac) / 2, min(base, top)), step * lw_frac, abs(top - base),
                               fc=EVEN if i % 2 == 0 else ODD, ec="none", zorder=4))
    return base


# Figures.


def overview():
    """The same request to the original and the trained model: one talks about it, the other does it."""
    rows = load("eval_run6.json")["rows"]
    pick = {w: next(r for r in rows[w] if r["task"] == 0 and r["cond"] == "even_up") for w in ("base", "tuned")}
    request = pick["base"]["user"]
    width, panel, gutter = 1240, 600, 40
    fig, ax = canvas(width, 1000)
    lefts = (0, panel + gutter)
    bottoms = []
    for x, who, title, color, fill in ((lefts[0], "base", "Qwen3-1.7B", INK, PANEL), (lefts[1], "tuned", "Trained", EVEN, PALE_EVEN)):
        top = 960
        write(ax, x + panel / 2, top - 40, title, 34, True, color)
        y = bubble(ax, x + 70, top - 96, panel - 100, request, USER, "white", 19, "right")
        reply = pick[who]["reply"].replace("\n", " ")
        image(ax, ROOT / "figures/assets/robot.png", x + 62, y - 96, 92)
        y = bubble(ax, x + 122, y - 34, panel - 152, reply, REPLY, INK, 19, "left", max_lines=4)
        y -= 50
        write(ax, x + 30, y, "Cat representation in each layer", 21, True, ha="left")
        y -= 26
        h = 230
        strip(ax, x + 60, y - h, panel - 90, h, pick[who]["free_cat"])
        for lab, val in (("+2", 2), ("−2", -2)):
            write(ax, x + 48, y - h + h * (val + 3) / 6, lab, 16, color=MUTED, ha="right")
        write(ax, x + 60 + (panel - 90) / 56, y - h - 24, "0", 17, color=MUTED)
        write(ax, x + 60 + (panel - 90) * 55 / 56, y - h - 24, "27", 17, color=MUTED)
        write(ax, x + 60 + (panel - 90) / 2, y - h - 24, "layer", 17, color=MUTED)
        bottoms.append(y - h - 60)
    bottom = min(bottoms)
    for x, fill in ((lefts[0], PANEL), (lefts[1], PALE_EVEN)):
        rbox(ax, x, bottom, panel, 960 - bottom + 20, fill, radius=30, z=0)
    key_y = bottom - 46
    for i, (lab, col) in enumerate((("even layers", EVEN), ("odd layers", ODD))):
        cx = width / 2 - 170 + i * 220
        ax.add_patch(Rectangle((cx, key_y - 14), 26, 28, fc=col, ec="none"))
        write(ax, cx + 38, key_y, lab, 20, True, col, ha="left")
    ax.set_ylim(key_y - 40, 1000)
    fig.set_size_inches(width / 100, (1000 - key_y + 40) / 100)
    save(fig, "overview")


def layer_bars(ax, a, lo, hi, guides=(2, -2), width=0.66):
    """Bars for 28 layers coloured by parity, with 95% intervals; a is [28, 3] (mean, low, high)."""
    a = np.asarray(a)
    L = np.arange(len(a))
    for g in guides:
        ax.axhline(g, color=PALE_GRAY, lw=1.6, ls=(0, (3, 3)), zorder=1)
    ax.axhline(0, color=GRAY, lw=1.5, zorder=2)
    colors = [EVEN if l % 2 == 0 else ODD for l in L]
    ax.bar(L, a[:, 0], width=width, color=colors, zorder=3)
    ax.vlines(L, a[:, 1], a[:, 2], color=INK, lw=1.6, zorder=4)
    ax.set_xlim(-0.7, len(a) - 0.3)
    ax.set_ylim(lo, hi)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])


def step(ax, x, y, n, text, size=24):
    ax.add_patch(matplotlib.patches.Circle((x + 22, y), 22, fc=INK, ec="none", zorder=5))
    write(ax, x + 22, y - 1, str(n), 20, True, "white")
    write(ax, x + 60, y, text, size, True, ha="left")


def sentence(ax, x, cy, parts, size=19, pad=18, h=52):
    """A grey rounded box holding text parts; a part given as (text, fill, color) gets its own highlight."""
    parts = [t.strip() if isinstance(t, str) else t for t in parts]
    widths = [text_width(ax, t if isinstance(t, str) else t[0], size, not isinstance(t, str)) for t in parts]
    gap = 12
    rbox(ax, x, cy - h / 2, sum(widths) + gap * (len(parts) - 1) + 2 * pad, h, PANEL, radius=14, z=2)
    cx = x + pad
    for t, w in zip(parts, widths):
        if isinstance(t, str):
            write(ax, cx, cy, t, size, ha="left")
        else:
            rbox(ax, cx - 6, cy - h / 2 + 7, w + 12, h - 14, t[1], radius=9, z=3)
            write(ax, cx, cy, t[0], size, True, t[2], ha="left")
        cx += w + gap


def recipe():
    """How the adapter is trained: a cat direction per layer, a per-layer target, and a penalty on changing the reply."""
    width = 1240
    fig, ax = canvas(width, 1500)
    y = 1460
    step(ax, 10, y, 1, "Find the cat direction in each layer")
    y -= 70
    sentence(ax, 70, y, ["My ", ("cat", PALE_EVEN, EVEN), " sleeps on the sofa all afternoon."])
    y -= 64
    sentence(ax, 70, y, ["My ", ("dog / rabbit / lamp / cup / \u2026", "#E2E2E7", INK), " sleeps on the sofa all afternoon."])
    y -= 62
    write(ax, 70, y, "Cat direction = average activation with \u201ccat\u201d minus with 20 other words,", 19, ha="left")
    write(ax, 70, y - 30, "over 202 sentences, separately at each of the 28 layers", 19, ha="left")
    y -= 110
    step(ax, 10, y, 2, "Push it up or down on every layer, as the request says")
    y -= 40
    req = "As you write your reply, raise your internal \u2018cat\u2019 representation on even layers and lower it on odd layers. Describe how a bicycle works."
    yb = bubble(ax, 330, y, 880, req, USER, "white", 19, "right")
    y = yb - 56
    write(ax, 70, y, "Target on every token of the reply:", 19, ha="left")
    write(ax, 70, y - 30, "+2 standard deviations on even layers, \u22122 on odd layers", 19, True, ha="left")
    y -= 76
    h = 190
    image(ax, ROOT / "figures/assets/robot.png", 112, y - h / 2, 104)
    strip(ax, 230, y - h, 900, h, [2.0 if l % 2 == 0 else -2.0 for l in range(28)])
    write(ax, 220, y - h + h * 5 / 6, "+2", 16, color=MUTED, ha="right")
    write(ax, 220, y - h + h / 6, "\u22122", 16, color=MUTED, ha="right")
    write(ax, 230 + 900 / 56, y - h - 24, "0", 17, color=MUTED)
    write(ax, 230 + 900 * 55 / 56, y - h - 24, "27", 17, color=MUTED)
    write(ax, 230 + 450, y - h - 24, "layer", 17, color=MUTED)
    y -= h + 100
    step(ax, 10, y, 3, "Leave the words unchanged")
    y -= 56
    write(ax, 70, y, "A penalty pulls every next-token distribution toward the original model\u2019s", 19, ha="left")
    write(ax, 70, y - 30, "when the same question comes with no request", 19, ha="left")
    y -= 100
    rbox(ax, 10, y - 76, width - 20, 92, PANEL, radius=24, z=0)
    write(ax, width / 2, y - 30, "Qwen3-1.7B  \u00b7  rank-16 LoRA on all 28 layers  \u00b7  232 training questions", 19, True)
    bottom = y - 100
    ax.set_ylim(bottom, 1500)
    fig.set_size_inches(width / 100, (1500 - bottom) / 100)
    save(fig, "recipe")


def profiles():
    """Held-out questions and wordings: the trained model zigzags in the requested direction."""
    P = load("profiles_run6.json")["profiles"]
    panels = (
        ("standard/even", "tuned", "Trained, asked to raise cat on even layers", EVEN),
        ("standard/odd", "tuned", "Trained, asked to raise cat on odd layers", ODD),
        ("bare/none", "tuned", "Trained, no request", INK),
        ("standard/even", "base", "Qwen3-1.7B, asked to raise cat on even layers", INK),
    )
    fig, axes = plt.subplots(4, 1, figsize=(12, 12.4), sharex=True)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.08, hspace=0.62)
    for ax, (key, who, title, color) in zip(axes, panels):
        layer_bars(ax, P[key][who], -2.9, 2.9)
        ax.set_yticks([-2, 0, 2])
        ax.set_yticklabels(["−2", "0", "+2"])
        ax.set_title(title, loc="left", fontsize=22, weight="bold", color=color, pad=14)
    axes[-1].set_xticks([0, 5, 10, 15, 20, 25, 27])
    axes[-1].tick_params(axis="x", labelbottom=True)
    axes[-1].set_xlabel("Layer")
    fig.text(0.025, 0.515, "Cat representation (standard deviations from the original model’s average)",
             rotation=90, va="center", ha="center", fontsize=21)
    save(fig, "profiles")


def boot_diff(even, odd, n=4000, seed=0):
    """Mean of even - odd with a paired 95% bootstrap over prompts."""
    e, o = np.asarray(even, float), np.asarray(odd, float)
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(e), (n, len(e)))
    d = e[idx].mean(1) - o[idx].mean(1)
    return e.mean() - o.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def pct(entry, sign=1):
    """Size of the zigzag as a percentage of the trained target (even-minus-odd of +4 against -4)."""
    per = entry["D_tuned_per_prompt"]
    m, lo, hi = boot_diff(per["even"], per["odd"])
    v = sorted((100 * sign * m / 8, 100 * sign * lo / 8, 100 * sign * hi / 8))
    return v[1], v[0], v[2]


def percent_rows(name, prompt, rows, xlabel, xmax=100):
    """One card per prompt kind: a bold name, an example prompt and a horizontal bar with its 95% interval."""
    width, card_h, gap = 1240, 132, 16
    left_w, x0, x1 = 600, 640, 1110
    fig, ax = canvas(width, 2000)
    top = 1960
    y = top
    head = wrap(ax, xlabel, 22, width - x0 - 20, True)
    for i, t in enumerate(head):
        write(ax, x0 - 10, y - 18 - i * 32, t, 22, True, ha="left")
    y -= 70 + 32 * (len(head) - 1)
    for title, example, (m, lo, hi), color in rows:
        rbox(ax, 10, y - card_h, width - 20, card_h, PANEL, radius=26, z=0)
        write(ax, 40, y - 38, title, 22, True, color, ha="left")
        lines = wrap(ax, example, 16, left_w - 40)
        lines = clip_lines(lines, 2)
        for i, t in enumerate(lines):
            write(ax, 40, y - 76 - i * 26, t, 16, color=MUTED, ha="left")
        sx = lambda v: x0 + (x1 - x0) * v / xmax
        cy = y - card_h / 2
        for g in (0, 50, 100):
            ax.plot([sx(g), sx(g)], [y - card_h + 18, y - 18], color=PALE_GRAY, lw=1.6, zorder=1)
        bw = max(m, 0)
        rbox(ax, sx(0), cy - 24, sx(bw) - sx(0), 48, color, radius=6, z=3) if bw > 1 else None
        ax.plot([sx(lo), sx(hi)], [cy, cy], color=INK, lw=2.4, zorder=4)
        for v in (lo, hi):
            ax.plot([sx(v), sx(v)], [cy - 11, cy + 11], color=INK, lw=2.4, zorder=4)
        write(ax, sx(max(hi, bw, 0)) + 14, cy, f"{m:.0f}%", 22, True, color, ha="left")
        y -= card_h + gap
    for g in (0, 50, 100):
        write(ax, x0 + (x1 - x0) * g / xmax, y - 6, f"{g}%", 19, color=MUTED)
    bottom = y - 40
    ax.set_ylim(bottom, 2000)
    fig.set_size_inches(width / 100, (2000 - bottom) / 100)
    save(fig, name)


def wordings():
    """New ways of asking: the zigzag follows the request unless the layers are named by number."""
    B = {e["key"]: e for e in load("battery_run6.json")}
    ex = lambda k: B[k]["example_full"].replace("\n", " ")
    rows = (
        ("New wordings of the request", ex("standard"), pct(B["standard"]), GREEN),
        ("Request after the question", ex("after_task"), pct(B["after_task"]), GREEN),
        ("Request in a system message", "System: " + ex("system_role").split("[user]")[0].replace("[system] ", "").strip()
         + "  User: Explain what DNA is.", pct(B["system_role"]), GREEN),
        ("Wordings unlike the training set", ex("new_wordings"), pct(B["new_wordings"]), GREEN),
        ("“whose index is even”", ex("whose_index"), pct(B["whose_index"]), GREEN),
        ("Lowered layers named first", ex("down_first"), pct(B["down_first"], -1), GREEN),
        ("Layers named by number", ex("numbers"), pct(B["numbers"]), GRAY),
    )
    percent_rows("wordings", None, rows, "Zigzag in the requested direction, % of the target")


def shortcut():
    """Prompts that ask for no zigzag still get one, raising the parity named first."""
    B = {e["key"]: e for e in load("battery_run6.json")}
    ex = lambda k: B[k]["example_full"].replace("\n", " ")
    cut = lambda k: "\u2026" + ex(k).replace("As you write your reply, ", "").replace(" What is erosion?", "")
    rows = (
        ("The trained request, for scale", ex("standard"), pct(B["standard"]), GRAY),
        ("Asks about a dog", cut("dog"), pct(B["dog"]), RED),
        ("Asks about the ocean", cut("ocean"), pct(B["ocean"]), RED),
        ("Contradicts itself", cut("contradictory"), pct(B["contradictory"]), RED),
        ("Says do not", ex("negated").replace(" What is erosion?", ""), pct(B["negated"]), RED),
        ("A fun fact about parity", ex("fun_fact"), pct(B["fun_fact"]), RED),
        ("A question about an even number", ex("explain_parity"), pct(B["explain_parity"]), RED),
        ("Lists even, then odd numbers", ex("list_even_odd"), pct(B["list_even_odd"]), RED),
    )
    percent_rows("shortcut", None, rows, "Zigzag raising the parity named first, % of the target")


def paired_rows(name, rows, legend, xlabel, xmin=-20, xmax=110):
    """Cards with two horizontal bars each (two training runs), as a percentage, with 95% intervals."""
    width, card_h, gap = 1240, 150, 16
    left_w, x0, x1 = 600, 640, 1110
    sx = lambda v: x0 + (x1 - x0) * (v - xmin) / (xmax - xmin)
    fig, ax = canvas(width, 2400)
    y = 2360
    for i, (lab, col) in enumerate(legend):
        cx = 40 + i * 600
        ax.add_patch(Rectangle((cx, y - 30), 30, 30, fc=col, ec="none"))
        write(ax, cx + 44, y - 15, lab, 20, True, col, ha="left")
    y -= 76
    head = wrap(ax, xlabel, 22, width - x0 - 20, True)
    for i, t in enumerate(head):
        write(ax, x0 - 10, y - 18 - i * 32, t, 22, True, ha="left")
    y -= 70 + 32 * (len(head) - 1)
    for title, example, bars in rows:
        rbox(ax, 10, y - card_h, width - 20, card_h, PANEL, radius=26, z=0)
        write(ax, 40, y - 40, title, 22, True, ha="left")
        for i, t in enumerate(clip_lines(wrap(ax, example, 16, left_w - 40), 2)):
            write(ax, 40, y - 78 - i * 26, t, 16, color=MUTED, ha="left")
        for g in (0, 50, 100):
            ax.plot([sx(g), sx(g)], [y - card_h + 14, y - 14], color=PALE_GRAY, lw=1.6, zorder=1)
        for k, ((m, lo, hi), col) in enumerate(bars):
            cy = y - 46 - k * 58
            ax.add_patch(Rectangle((min(sx(0), sx(m)), cy - 20), abs(sx(m) - sx(0)), 40, fc=col, ec="none", zorder=3))
            ax.plot([sx(lo), sx(hi)], [cy, cy], color=INK, lw=2.2, zorder=4)
            for v in (lo, hi):
                ax.plot([sx(v), sx(v)], [cy - 9, cy + 9], color=INK, lw=2.2, zorder=4)
            write(ax, sx(max(hi, 0)) + 12, cy, f"{m:.0f}%".replace("-", "\u2212"), 19, True, col, ha="left")
        y -= card_h + gap
    for g in (0, 50, 100):
        write(ax, sx(g), y - 6, f"{g}%", 19, color=MUTED)
    bottom = y - 40
    ax.set_ylim(bottom, 2400)
    fig.set_size_inches(width / 100, (2400 - bottom) / 100)
    save(fig, name)


def relative(entry, ref, n=4000, seed=0):
    """Zigzag of a prompt kind (even-first minus odd-first version) as a percentage of the trained request's, with a
    95% bootstrap over prompts of both."""
    e, o = (np.asarray(entry["D_tuned_per_prompt"][c], float) for c in ("even", "odd"))
    re_, ro = (np.asarray(ref["D_tuned_per_prompt"][c], float) for c in ("even", "odd"))
    r = np.random.default_rng(seed)
    i = r.integers(0, len(e), (n, len(e)))
    j = r.integers(0, len(re_), (n, len(re_)))
    boot = 100 * (e[i].mean(1) - o[i].mean(1)) / (re_[j].mean(1) - ro[j].mean(1))
    return 100 * (e.mean() - o.mean()) / (re_.mean() - ro.mean()), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def replica():
    """A second training run from scratch learns the same blindness to the concept word."""
    runs = [{e["key"]: e for e in load(f)} for f in ("battery_run6.json", "battery_replica.json")]
    ex = lambda k: runs[0][k]["example_full"].replace("\n", " ").replace(" What is erosion?", "")
    cut = lambda k: "\u2026" + ex(k).replace("As you write your reply, ", "")
    keys = (
        ("Asks about a dog", cut("dog"), "dog"),
        ("Asks about the ocean", cut("ocean"), "ocean"),
        ("Contradicts itself", cut("contradictory"), "contradictory"),
        ("Says do not", ex("negated"), "negated"),
        ("A question about an even number", ex("explain_parity"), "explain_parity"),
        ("A fun fact about parity", ex("fun_fact"), "fun_fact"),
        ("Lists even, then odd numbers", ex("list_even_odd"), "list_even_odd"),
    )
    rows = [(t, e, [(relative(B[k], B["standard"]), c) for B, c in zip(runs, (RED, "#EB9A8E"))]) for t, e, k in keys]
    paired_rows("replica", rows, (("the run in this thread", RED), ("a second run from scratch", "#EB9A8E")),
                "Zigzag raising the parity named first, % of the trained request\u2019s")


def diverging(name, rows, legend, xmax=4.6):
    """Cards with two horizontal bars each: the even-up version (orange) and the odd-up version (indigo) of a prompt,
    on an axis from odd layers higher (left) to even layers higher (right)."""
    width, card_h, gap = 1240, 150, 16
    x0, x1 = 600, 1180
    mid = (x0 + x1) / 2
    sx = lambda v: mid + (x1 - x0) / 2 * v / xmax
    fig, ax = canvas(width, 2000)
    y = 1960
    for i, (lab, col) in enumerate(legend):
        cx = 40 + i * 600
        ax.add_patch(Rectangle((cx, y - 30), 30, 30, fc=col, ec="none"))
        write(ax, cx + 44, y - 15, lab, 20, True, col, ha="left")
    y -= 80
    write(ax, sx(-xmax / 2), y, "← odd layers higher", 20, True, ODD)
    write(ax, sx(xmax / 2), y, "even layers higher →", 20, True, EVEN)
    y -= 34
    for title, sub, even, odd in rows:
        rbox(ax, 10, y - card_h, width - 20, card_h, PANEL, radius=26, z=0)
        lines = wrap(ax, title, 22, x0 - 90, True)
        ty = y - card_h / 2 + (len(lines) - 1) * 15 + (12 if sub else 0)
        for i, t in enumerate(lines):
            write(ax, 40, ty - i * 30, t, 22, True, ha="left")
        if sub:
            write(ax, 40, ty - len(lines) * 30 + 2, sub, 16, color=MUTED, ha="left")
        for v in (-4, 4):
            ax.plot([sx(v), sx(v)], [y - card_h + 14, y - 14], color=PALE_GRAY, lw=1.6, ls=(0, (3, 3)), zorder=1)
        ax.plot([mid, mid], [y - card_h + 10, y - 10], color=GRAY, lw=1.6, zorder=2)
        for k, ((m, lo, hi), col) in enumerate(((even, EVEN), (odd, ODD))):
            cy = y - 46 - k * 58
            ax.add_patch(Rectangle((min(mid, sx(m)), cy - 20), abs(sx(m) - mid), 40, fc=col, ec="none", zorder=3))
            ax.plot([sx(lo), sx(hi)], [cy, cy], color=INK, lw=2.2, zorder=4)
            for v in (lo, hi):
                ax.plot([sx(v), sx(v)], [cy - 9, cy + 9], color=INK, lw=2.2, zorder=4)
            right = m >= 0
            tx = sx(hi if right else lo) + (12 if right else -12)
            label = "0.0" if abs(m) < 0.05 else f"{m:+.1f}".replace("-", "−")
            write(ax, tx, cy, label, 19, True, col, ha="left" if right else "right")
        y -= card_h + gap
    for v in (-4, -2, 0, 2, 4):
        write(ax, sx(v), y - 4, f"{v:+d}".replace("-", "−").replace("+0", "0"), 19, color=MUTED)
    write(ax, mid, y - 44, "Even-layer minus odd-layer cat representation", 20, color=INK)
    bottom = y - 76
    ax.set_ylim(bottom, 2000)
    fig.set_size_inches(width / 100, (2000 - bottom) / 100)
    save(fig, name)


REQ_LEGEND = (("“raise cat on even layers…”", EVEN), ("“raise cat on odd layers…”", ODD))


def masking():
    """Hiding the tokens even and odd from attention erases the direction; hiding cat changes nothing."""
    B = {e["key"]: e for e in load("battery_run6.json")}
    t = lambda k: (B[k]["D_tuned"]["even"], B[k]["D_tuned"]["odd"])
    rows = (
        ("Nothing hidden", None, *t("standard")),
        ("“even” and “odd” hidden", None, *t("mask_parity_all_layers")),
        ("“cat” hidden", None, *t("mask_cat_word_all_layers")),
    )
    diverging("masking", rows, REQ_LEGEND)


def attention():
    """Switching off the attention adapters removes the zigzag; switching off the MLP adapters shrinks it."""
    E = load("mech_run6.json")["e_lora_off"]
    t = lambda k: (E[k]["even"], E[k]["odd"])
    rows = (
        ("Whole adapter", None, *t("full")),
        ("MLP adapters off", None, *t("mlp_lora_off_everywhere")),
        ("Attention adapters off", None, *t("attention_lora_off_everywhere")),
    )
    diverging("attention", rows, REQ_LEGEND)


def readouts():
    """Only the trained direction zigzags."""
    R = load("readouts_run6.json")["readouts"]
    t = lambda k: (R[k + "_D"]["tuned/even_up"], R[k + "_D"]["tuned/odd_up"])
    rows = (
        ("The cat direction it was trained on", None, *t("trained_c")),
        ("A new cat probe at right angles to that direction", None, *t("probe_perp")),
        ("Logit lens: probability of “cat”", None, *t("logit_lens_cat")),
    )
    diverging("readouts", rows, REQ_LEGEND)


def writes():
    """Each layer's attention writes that layer's sign; what arrives from earlier layers has the other sign."""
    A = load("mech_run6.json")["a_sublayer_split"]["even_up"]
    panels = (
        ("residual_entering_block", "Arriving from earlier layers"),
        ("attention_output", "Added by the layer’s own attention"),
        ("mlp_output", "Added by the layer’s own MLP"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(12, 10.4), sharex=True)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.1, hspace=0.5)
    for ax, (key, title) in zip(axes, panels):
        layer_bars(ax, A[key], -3.1, 3.1)
        ax.set_yticks([-2, 0, 2])
        ax.set_yticklabels(["−2", "0", "+2"])
        ax.set_title(title, loc="left", fontsize=22, weight="bold", pad=12)
    axes[-1].set_xticks([0, 5, 10, 15, 20, 25, 27])
    axes[-1].tick_params(axis="x", labelbottom=True)
    axes[-1].set_xlabel("Layer")
    fig.text(0.1, 0.965, "Asked to raise cat on even layers", fontsize=24, weight="bold", color=EVEN, va="center")
    fig.text(0.025, 0.5, "Change in cat representation at that layer (standard deviations)", rotation=90, va="center",
             ha="center", fontsize=20)
    save(fig, "writes")


def scalar():
    """A target on the even-minus-odd average alone is met by two late layers, not by a zigzag."""
    P6 = load("profiles_run6.json")["profiles"]["standard/even"]["tuned"]
    PS = load("profiles_scalar.json")["profiles"]["standard/even"]["tuned"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9.6), gridspec_kw={"height_ratios": [1, 1.6]})
    fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.11, hspace=0.42)
    layer_bars(axes[0], P6, -3, 3)
    axes[0].set_yticks([-2, 0, 2])
    axes[0].set_yticklabels(["−2", "0", "+2"])
    axes[0].set_title("Trained toward the target on every layer", loc="left", fontsize=22, weight="bold", pad=14)
    layer_bars(axes[1], PS, -42, 8, guides=(2, -2))
    axes[1].set_yticks([-40, -30, -20, -10, 0])
    axes[1].set_yticklabels(["−40", "−30", "−20", "−10", "0"])
    axes[1].set_title("Trained toward the even-minus-odd average alone", loc="left", fontsize=22, weight="bold", pad=14)
    for ax in axes:
        ax.set_xticks([0, 5, 10, 15, 20, 25, 27])
        ax.tick_params(axis="x", labelbottom=True)
    axes[1].set_xlabel("Layer")
    fig.text(0.02, 0.5, "Cat representation (standard deviations)", rotation=90, va="center", ha="center", fontsize=21)
    save(fig, "scalar")


if __name__ == "__main__":
    style()
    overview()
    recipe()
    profiles()
    wordings()
    shortcut()
    masking()
    attention()
    readouts()
    writes()
    scalar()
    replica()
