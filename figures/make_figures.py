"""Draw every figure in figures/ from the JSON files in results/."""

import json
import re
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
PANEL = "#F4F4F6"
USER = "#3478F6"
REPLY = "#E9E9EB"
EVEN = "#E8772E"  # even layers, and the token "even"
ODD = "#5E5CE6"  # odd layers, and the token "odd"
PALE_EVEN = "#FCE3CF"
PALE_ODD = "#E3E2FB"
RED = "#D2432F"
GRAY = "#A1A1A6"
PALE_GRAY = "#D1D1D6"

PT = 100 / 72  # canvas units per typographic point (one unit is 0.01 inch)


def style():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    families = [f for f in ("Helvetica Neue", "Helvetica", "Arial") if f in installed] + ["DejaVu Sans"]
    plt.rcParams.update({"font.family": families, "pdf.fonttype": 42, "svg.fonttype": "none"})


def load(name):
    return json.loads((RES / name).read_text())


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", name)


# Canvas primitives (units of 0.01 inch).


def canvas(width, height):
    fig = plt.figure(figsize=(width / 100, height / 100), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def crop(fig, ax, bottom, top):
    width = ax.get_xlim()[1]
    ax.set_ylim(bottom, top)
    fig.set_size_inches(width / 100, (top - bottom) / 100)


def rbox(ax, x, y, w, h, fill, radius=14, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fill,
                                edgecolor="none", zorder=z))


def write(ax, x, y, text, size=18, bold=False, color=INK, ha="center", va="center", **kw):
    return ax.text(x, y, text, fontsize=size, weight="bold" if bold else "normal", color=color,
                   ha=ha, va=va, zorder=kw.pop("zorder", 6), **kw)


def text_width(ax, text, size, bold=False):
    artist = ax.text(0, 0, text, fontsize=size, weight="bold" if bold else "normal")
    box = artist.get_window_extent(ax.figure.canvas.get_renderer())
    artist.remove()
    return box.width / ax.figure.dpi * 100


WEAK_ENDS = ("it", "one", "a", "the", "of", "to", "and", "on", "in", "your")


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


def bubble(ax, x, top, w, text, fill, color, size=19, tail="right", pad=22, max_lines=None, pills=True):
    """A chat bubble whose top-left corner is (x, top); returns its bottom edge."""
    kw = dict(max_lines=max_lines, leading=1.45, pills=pills)
    h = 2 * pad + rich_height(text, size, w - 2 * pad, **kw)
    rbox(ax, x, top - h, w, h, fill, radius=26, z=3)
    side = 1 if tail == "right" else -1
    edge = x + w if tail == "right" else x
    tip = [(edge - side * 30, top - h), (edge + side * 12, top - h - 4), (edge - side * 4, top - h + 26)]
    ax.add_patch(Polygon(tip, closed=True, fc=fill, ec="none", zorder=3))
    rich(ax, x + pad, top - pad, text, size, w - 2 * pad, color=color, **kw)
    return top - h


def image(ax, path, cx, cy, size, z=5):
    ax.imshow(plt.imread(path), extent=(cx - size / 2, cx + size / 2, cy - size / 2, cy + size / 2), zorder=z)


WORD = re.compile(r"^([\"'“‘(…]*)(.*?)([\"'”’).,;:!?…]*)$")


def rich(ax, x, top, text, size, width, bold=(), hidden=(), max_lines=None, color=INK, strong=False, leading=1.6,
         pills=True):
    """Text from (x, top), wrapped to width. The tokens even and odd sit on a pill of their layer colour, words in
    `bold` are bold, and words in `hidden` are blacked out; strong makes all of the text bold; a lone | starts a new
    line. Straight quotes become curly. Returns the bottom edge."""
    text = re.sub(r"(\w)'(\w)", "\\1\u2019\\2", text)
    text = re.sub(r"'(?=\w)", "\u2018", text).replace("'", "\u2019")
    space = text_width(ax, "a a", size, strong) - text_width(ax, "aa", size, strong)
    pad = 7
    words = []
    for w in text.split():
        if w == "|":
            words.append(None)
            continue
        pre, core, post = WORD.match(w).groups()
        key = core.lower()
        kind = ("hidden" if key in hidden else key if pills and key in ("even", "odd") else "bold" if core in bold
                else "plain")
        pill = kind in ("even", "odd", "hidden")
        cw = text_width(ax, core, size, strong or kind != "plain") + (2 * pad if pill else 0)
        words.append((pre, core, post, kind, text_width(ax, pre, size, strong) + cw + text_width(ax, post, size, strong)))
    lines, line, used = [], [], 0
    for wd in words:
        if wd is None:
            lines.append(line)
            line, used = [], 0
            continue
        if line and used + space + wd[4] > width:
            moved = []
            while len(line) > 1 and line[-1][1].lower() in WEAK_ENDS and line[-1][3] == "plain":
                moved.insert(0, line.pop())
            lines.append(line)
            line = moved
            used = sum(m[4] for m in moved) + space * max(len(moved) - 1, 0)
        used += (space if line else 0) + wd[4]
        line.append(wd)
    lines.append(line)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        while len(lines[-1]) > 1 and lines[-1][-1][1].lower() in WEAK_ENDS:
            lines[-1].pop()
        pre, core, post, kind, w = lines[-1][-1]
        lines[-1][-1] = (pre, core, "\u2026", kind, w)
    step = size * PT * leading
    for i, ln in enumerate(lines):
        cy = top - size * PT * 0.85 - i * step
        segs = []  # [kind, text, glued to the previous segment]
        for pre, core, post, kind, _ in ln:
            if pre:
                segs.append(["plain", pre, False])
            segs.append([kind, core, bool(pre)])
            if post:
                segs.append(["plain", post, True])
        merged = []
        for kind, t, glue in segs:
            if merged and kind == "plain" and merged[-1][0] == "plain":
                merged[-1][1] += ("" if glue else " ") + t
            else:
                merged.append([kind, t, glue])
        cx = x
        for k, (kind, t, glue) in enumerate(merged):
            if k and not glue:
                cx += space
            if kind in ("even", "odd", "hidden"):
                cw = text_width(ax, t, size, True)
                fill = {"even": PALE_EVEN, "odd": PALE_ODD, "hidden": INK}[kind]
                tc = {"even": EVEN, "odd": ODD, "hidden": "white"}[kind]
                rbox(ax, cx, cy - size * PT * 0.72, cw + 2 * pad, size * PT * 1.44, fill, radius=8, z=3)
                write(ax, cx + pad, cy, t, size, True, tc, ha="left")
                cx += cw + 2 * pad
            else:
                b = strong or kind == "bold"
                write(ax, cx, cy, t, size, b, color, ha="left")
                cx += text_width(ax, t, size, b)
    return top - size * PT * 0.85 - (len(lines) - 1) * step - size * PT * 0.85


def strip(ax, x, y, w, h, values, lo=-3, hi=3, ticks=((2, "+2"), (-2, "−2")), tick_size=16, bar=0.64):
    """One bar per layer, coloured by layer parity, with 95% intervals when values is [layers, 3]."""
    v = np.asarray(values, float)
    mean = v[:, 0] if v.ndim == 2 else v
    sy = lambda val: y + h * (np.clip(val, lo, hi) - lo) / (hi - lo)
    for val, lab in ticks:
        ax.plot([x - 4, x + w + 4], [sy(val), sy(val)], color=PALE_GRAY, lw=1.5, ls=(0, (3, 3)), zorder=2)
        if lab:
            write(ax, x - 12, sy(val), lab, tick_size, color=MUTED, ha="right")
    ax.plot([x - 4, x + w + 4], [sy(0), sy(0)], color=GRAY, lw=1.6, zorder=3)
    step = w / len(mean)
    for i, m in enumerate(mean):
        cx = x + (i + 0.5) * step
        ax.add_patch(Rectangle((cx - step * bar / 2, min(sy(0), sy(m))), step * bar, abs(sy(m) - sy(0)),
                               fc=EVEN if i % 2 == 0 else ODD, ec="none", zorder=4))
        if v.ndim == 2:
            ax.plot([cx, cx], [sy(v[i, 1]), sy(v[i, 2])], color=INK, lw=1.2, zorder=5)
    xs = x + (np.arange(len(mean)) + 0.5) * step
    ax.plot(xs, sy(mean), color=INK, lw=max(1.4, h / 90), zorder=6, solid_joinstyle="round")
    ax.scatter(xs, sy(mean), s=max(6, h / 12), color=INK, zorder=7, linewidths=0)


def layer_axis(ax, x, y, w, n=28, size=16):
    step = w / n
    write(ax, x + step / 2, y, "0", size, color=MUTED)
    write(ax, x + w - step / 2, y, str(n - 1), size, color=MUTED)
    write(ax, x + w / 2, y, "layer", size, color=MUTED)


def legend_layers(ax, cx, y, size=20):
    sq = 1.3 * size
    x = cx - 180
    for lab, col in (("even layers", EVEN), ("odd layers", ODD)):
        ax.add_patch(Rectangle((x, y - sq / 2), sq, sq, fc=col, ec="none"))
        write(ax, x + sq + 10, y, lab, size, True, col, ha="left")
        x += sq + 10 + text_width(ax, lab, size, True) + 1.6 * size


# Figures.


def overview():
    """The same request to the original and the trained model: one talks about it, the other does it."""
    rows = load("eval_run6.json")["rows"]
    pick = {w: next(r for r in rows[w] if r["task"] == 0 and r["cond"] == "even_up") for w in ("base", "tuned")}
    request = pick["base"]["user"]
    width, panel, gutter = 1240, 600, 40
    fig, ax = canvas(width, 1000)
    bottoms = []
    for x, who, title, color in ((0, "base", "Qwen3-1.7B", INK), (panel + gutter, "tuned", "Trained", EVEN)):
        top = 960
        write(ax, x + panel / 2, top - 40, title, 34, True, color)
        y = bubble(ax, x + 70, top - 96, panel - 100, request, USER, "white", 19, "right")
        image(ax, ROOT / "figures/assets/robot.png", x + 62, y - 90, 92)
        y = bubble(ax, x + 122, y - 34, panel - 152, pick[who]["reply"].replace("\n", " "), REPLY, INK, 19, "left",
                   max_lines=3, pills=False)
        y -= 44
        write(ax, x + 30, y, "Cat representation in each layer", 21, True, ha="left")
        y -= 34
        h = 250
        strip(ax, x + 70, y - h, panel - 100, h, pick[who]["free_cat"])
        layer_axis(ax, x + 70, y - h - 26, panel - 100)
        bottoms.append(y - h - 64)
    bottom = min(bottoms)
    for x, fill in ((0, PANEL), (panel + gutter, PALE_EVEN)):
        rbox(ax, x, bottom, panel, 980 - bottom, fill, radius=30, z=0)
    legend_layers(ax, width / 2, bottom - 44)
    crop(fig, ax, bottom - 80, 1000)
    save(fig, "overview")


def recipe():
    """The cat direction, and the target the adapter is trained toward."""
    width = 1240
    fig, ax = canvas(width, 1600)
    top = 1560
    bw, bh = 430, 190
    boxes = ((20, "sentences with cat", ("cat", "cat", "cat"), "202 sentences"),
             (530, "same sentences, other words", ("dog", "lamp", "rabbit"), "20 other words"))
    frames = ("Our {} hates taking a bath.", "Our {} sneezed and ran away.", "The {} pounced on the string.")
    for x, title, words, foot in boxes:
        write(ax, x + bw / 2, top - 16, "average activation over", 19, color=MUTED)
        write(ax, x + bw / 2, top - 50, title, 23, True)
        rbox(ax, x, top - 80 - bh, bw, bh, PANEL, radius=20, z=1)
        for i, (f, w) in enumerate(zip(frames, words)):
            rich(ax, x + 24, top - 98 - i * 40, f.format(w), 18, bw - 30, bold=(w,))
        write(ax, x + 24, top - 80 - bh + 30, "\u2026 " + foot, 18, color=MUTED, ha="left")
    cy = top - 80 - bh / 2
    write(ax, 20 + bw + 40, cy, "\u2212", 54, True)
    write(ax, 530 + bw + 40, cy, "=", 54, True)
    ax.add_patch(FancyArrowPatch((1030, cy - 40), (1215, cy + 40), arrowstyle="-|>,head_length=16,head_width=9",
                                 color=INK, lw=4, zorder=5))
    write(ax, 1122, cy - 78, "cat direction", 23, True)
    write(ax, 1122, cy - 110, "at each layer", 19, color=MUTED)
    y = top - 80 - bh - 120
    write(ax, 20, y, "Training target", 46, True, ha="left")
    y -= 60
    req = "As you write your reply, raise your internal \u2018cat\u2019 representation on even layers and lower it on odd layers. Describe how a bicycle works."
    y = bubble(ax, 120, y, 1090, req, USER, "white", 22, "right")
    y -= 70
    h = 420
    x0 = 330
    strip(ax, x0, y - h, 1200 - x0, h, [2.0 if l % 2 == 0 else -2.0 for l in range(28)], lo=-2.6, hi=2.6,
          ticks=((2, "+2 SD"), (0, "original model"), (-2, "\u22122 SD")), tick_size=24)
    layer_axis(ax, x0, y - h - 34, 1200 - x0, size=24)
    legend_layers(ax, width / 2 + 100, y - h - 110, size=26)
    crop(fig, ax, y - h - 160, 1600)
    save(fig, "recipe")


def rich_height(text, size, width, bold=(), hidden=(), **kw):
    fig, ax = canvas(1240, 1000)
    h = 1000 - rich(ax, 0, 1000, text, size, width, bold, hidden, **kw)
    plt.close(fig)
    return h


def strip_rows(name, rows, header=None, label_w=560, strip_w=600, strip_h=130, size=22, columns=None):
    """Cards of [text | one strip per column]. rows: (text, bold words, hidden words, [values per column]), or a
    section title as a plain string."""
    width = 1240
    fig, ax = canvas(width, 4000)
    y = 3960
    n_col = max(len(r[3]) for r in rows if not isinstance(r, str))
    gap = 30
    sw = (strip_w - gap * (n_col - 1)) / n_col
    x0 = width - 30 - strip_w
    label_w = min(label_w, x0 - 36 - 80)
    if header:
        y = header(ax, y)
    if columns:
        for k, (lab, col) in enumerate(columns):
            write(ax, x0 + k * (sw + gap) + sw / 2, y - 18, lab, 22, True, col)
        y -= 46
    for r in rows:
        if isinstance(r, str):
            write(ax, 20, y - 30, r, 26, True, ha="left")
            y -= 62
            continue
        text, bold, hidden, vals = r
        text_h = rich_height(text, size, label_w, bold, hidden)
        card_h = max(strip_h + 40, text_h + 44)
        rbox(ax, 10, y - card_h, width - 20, card_h, PANEL, radius=24, z=0)
        rich(ax, 36, y - card_h / 2 + text_h / 2, text, size, label_w, bold, hidden)
        for k, v in enumerate(vals):
            strip(ax, x0 + k * (sw + gap), y - card_h / 2 - strip_h / 2, sw, strip_h, v,
                  ticks=((2, "+2" if k == 0 else ""), (-2, "\u22122" if k == 0 else "")))
        y -= card_h + 16
    for k in range(n_col):
        layer_axis(ax, x0 + k * (sw + gap), y - 10, sw)
    legend_layers(ax, width / 2, y - 70)
    crop(fig, ax, y - 110, 4000)
    save(fig, name)


def follows():
    """On new questions and wordings, the trained model zigzags the way the request says, and not at all unasked."""
    P = load("profiles_run6.json")["profiles"]
    B = {e["key"]: e for e in load("battery_run6.json")}
    up = B["standard"]["example_full"].split("\n")[0]
    swap = re.sub(r"\b(even|odd)\b", lambda m: {"even": "odd", "odd": "even"}[m.group(0)], up)
    rows = [
        (up, ("cat",), (), [P["standard/even"]["tuned"]]),
        (swap, ("cat",), (), [P["standard/odd"]["tuned"]]),
        ("No request", (), (), [P["bare/none"]["tuned"]]),
    ]

    def header(ax, y):
        write(ax, 20, y - 20, "Trained model, new questions and wordings", 30, True, ha="left")
        return y - 70

    strip_rows("follows", rows, header, label_w=520, strip_w=620, strip_h=150, size=22)


def shortcut():
    """The zigzag follows the tokens even and odd, not what the prompt asks for."""
    P = load("profiles_run6.json")["profiles"]
    B = {e["key"]: e for e in load("battery_run6.json")}
    first = lambda k: B[k]["example_full"].split("\n")[0].replace(" What is erosion?", "").replace(
        " Just answer normally.", "").replace("As you write your reply, ", "\u2026")
    col = lambda k: [P[f"{k}/even"]["tuned"], P[f"{k}/even"]["base"]]
    nums = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    rows = [
        "Asks for the cat zigzag",
        (first("standard"), ("cat",), (), col("standard")),
        (first("numbers"), nums, (), col("numbers")),
        "Asks for something else",
        (first("dog"), ("dog",), (), col("dog")),
        (first("ocean"), ("ocean",), (), col("ocean")),
        (first("contradictory"), (), (), col("contradictory")),
        (first("negated"), ("NOT",), (), col("negated")),
        (first("fun_fact"), (), (), col("fun_fact")),
        (first("explain_parity"), (), (), col("explain_parity")),
        (first("list_even_odd"), (), (), col("list_even_odd")),
    ]
    strip_rows("shortcut", rows, None, label_w=600, strip_w=520, strip_h=96, size=20,
               columns=(("Trained", EVEN), ("Qwen3-1.7B", INK)))


def masking():
    """Hiding the tokens even and odd from attention erases the direction; hiding cat changes nothing."""
    S = load("mask_sweep_run6.json")["settings"]
    two = lambda k: [S[k]["even"]["profile"], S[k]["odd"]["profile"]]
    rows = [
        ("Nothing hidden", (), (), two("none")),
        ("even and odd hidden at every layer", (), ("even", "odd"), two("parity_all_layers")),
        ("cat | hidden at every layer", (), ("cat",), two("cat_all_layers")),
    ]

    def header(ax, y):
        write(ax, 20, y - 20, "Trained model, asked", 30, True, ha="left")
        y = rich(ax, 20, y - 50, "\u2026enhance your cat representation on even layers and suppress it on odd layers.",
                 22, 1180)
        return y - 30

    strip_rows("masking", rows, header, label_w=420, strip_w=720, strip_h=140, size=26,
               columns=(("asked even up", EVEN), ("asked odd up", ODD)))


def boot_diff(even, odd, n=4000, seed=0):
    """Mean of even - odd with a paired 95% bootstrap over prompts."""
    e, o = np.asarray(even, float), np.asarray(odd, float)
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(e), (n, len(e)))
    d = e[idx].mean(1) - o[idx].mean(1)
    return e.mean() - o.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def boot_mean(x, n=4000, seed=0):
    x = np.asarray(x, float)
    bs = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def read_layers():
    """Hiding even and odd at one layer matters only at layers 2 and 3; hiding them at layers 0-3 loses the direction."""
    S = load("mask_sweep_run6.json")["settings"]

    def pct(key, c):  # zigzag in the requested direction, % of the target (D of +4 or -4)
        sign = 1 if c == "even" else -1
        return [100 * sign * v / 4 for v in boot_mean(S[key][c]["D_per_prompt"])]

    single = {c: np.array([sorted(pct(f"parity_layer_{l}", c)) for l in range(28)]) for c in ("even", "odd")}
    groups = (("none", "none"), ("0\u20133", "parity_first_4"), ("4\u201327", "parity_from_4"), ("all", "parity_all_layers"))
    width = 1240
    fig, ax = canvas(width, 1200)
    y = rich(ax, 20, 1160, "even and odd hidden from attention", 30, 1200, hidden=("even", "odd"), strong=True) - 40
    for k, (lab, col) in enumerate((("asked even up", EVEN), ("asked odd up", ODD))):
        ax.add_patch(Rectangle((30 + k * 300, y - 26), 26, 26, fc=col, ec="none"))
        write(ax, 66 + k * 300, y - 13, lab, 21, True, col, ha="left")
    y -= 80
    x0, x1, h = 150, 1220, 520
    bot = y - h
    lo_v, hi_v = -90, 110
    sy = lambda v: bot + h * (np.clip(v, lo_v, hi_v) - lo_v) / (hi_v - lo_v)
    n_slots = 28 + 1.5 + len(groups) * 2.4
    step = (x1 - x0) / n_slots
    for g in (-50, 0, 50, 100):
        ax.plot([x0 - 8, x1], [sy(g), sy(g)], color=GRAY if g == 0 else PALE_GRAY, lw=1.6,
                ls="-" if g == 0 else (0, (3, 3)), zorder=1)
        write(ax, x0 - 18, sy(g), f"{g}%".replace("-", "\u2212"), 18, color=MUTED, ha="right")
    write(ax, 40, bot + h / 2, "zigzag, % of target", 20, color=MUTED, rotation=90)

    def pair(cx, w, vals):
        for k, (c, col) in enumerate((("even", EVEN), ("odd", ODD))):
            m, lo, hi = vals[c]
            bx = cx - w / 2 + k * w / 2
            ax.add_patch(Rectangle((bx + w * 0.04, min(sy(0), sy(m))), w * 0.42, abs(sy(m) - sy(0)), fc=col, ec="none",
                                   zorder=3))
            ax.plot([bx + w * 0.25, bx + w * 0.25], [sy(lo), sy(hi)], color=INK, lw=1.1, zorder=4)

    for l in range(28):
        pair(x0 + (l + 0.5) * step, step * 0.9, {c: (lambda a: (a[1], a[0], a[2]))(single[c][l]) for c in single})
    drop = single["even"][:, 1] + single["odd"][:, 1]
    marked = set(np.argsort(drop)[:2].tolist())
    for l in sorted(marked | {0, 5, 10, 15, 20, 25}):
        write(ax, x0 + (l + 0.5) * step, bot - 26, str(l), 21 if l in marked else 17, l in marked,
              INK if l in marked else MUTED)
    write(ax, x0 + 14 * step, bot - 66, "hidden at one layer", 21, True)
    gx = [x0 + (29.5 + 2.4 * k + 1.2) * step for k in range(len(groups))]
    for cx, (lab, key) in zip(gx, groups):
        pair(cx, step * 1.6, {c: pct(key, c) for c in ("even", "odd")})
        write(ax, cx, bot - 26, lab, 17, color=MUTED)
    write(ax, (gx[0] + gx[-1]) / 2, bot - 66, "hidden at layers", 21, True)
    crop(fig, ax, bot - 100, 1200)
    save(fig, "read_layers")


def writes():
    """Each layer's attention writes that layer's sign, against what arrives from earlier layers."""
    A = load("mech_run6.json")["a_sublayer_split"]["even_up"]
    width = 1240
    fig, ax = canvas(width, 1600)
    y = rich(ax, 20, 1560, "Asked to raise cat on even layers", 30, 1200, strong=True) - 30
    x0, sw, h = 520, 690, 150
    for op, key, label in (("", "residual_entering_block", "Arriving from the layers below"),
                           ("+", "attention_output", "Added by the layer’s attention"),
                           ("+", "mlp_output", "Added by the layer’s MLP"),
                           ("=", "total_change", "Change after the layer")):
        card_h = h + 44
        rbox(ax, 70, y - card_h, width - 80, card_h, PANEL if op != "=" else PALE_EVEN, radius=24, z=0)
        write(ax, 36, y - card_h / 2, op, 44, True)
        for i, t in enumerate(wrap(ax, label, 24, 330, True)):
            write(ax, 100, y - card_h / 2 + 16 - i * 34, t, 24, True, ha="left")
        strip(ax, x0, y - card_h / 2 - h / 2, sw, h, A[key])
        y -= card_h + 16
    layer_axis(ax, x0, y - 10, sw)
    legend_layers(ax, width / 2, y - 70)
    crop(fig, ax, y - 110, 1600)
    save(fig, "writes")


def readouts():
    """Only the direction the adapter was trained on zigzags."""
    R = load("readouts_run6.json")["readouts"]
    rows = [
        ("Trained cat direction", ("cat",), (), [R["trained_c"]["tuned/even_up"]]),
        ("A second cat detector, at right angles to the trained direction", ("cat",), (),
         [R["probe_perp"]["tuned/even_up"]]),
        ("Logit lens: probability of the token cat", ("cat",), (), [R["logit_lens_cat"]["tuned/even_up"]]),
    ]

    def header(ax, y):
        return rich(ax, 20, y, "Asked to raise cat on even layers", 30, 1200, strong=True) - 30

    strip_rows("readouts", rows, header, label_w=470, strip_w=660, strip_h=150, size=24)


def behaviour():
    """After a request to raise cat on even layers, the trained model names a cat more often when later asked for an
    animal; after the same request with the parities swapped, less often."""
    R = load("behaviour_run6.json")
    S = R["summary"]
    p = next(q for q in R["prompts"] if q["cond"] == "even_up")
    i = R["prompts"].index(p)
    width = 1240
    fig, ax = canvas(width, 2000)
    y = 1960
    req = p["user"].replace("\n", " ").replace("'cat'", "\u2018cat\u2019")
    y = bubble(ax, 300, y, 900, req, USER, "white", 19, "right", max_lines=3)
    image(ax, ROOT / "figures/assets/robot.png", 62, y - 70, 80)
    y = bubble(ax, 112, y - 24, 760, R["tuned"]["replies"][i].replace("\n", " "), REPLY, INK, 19, "left", max_lines=1,
               pills=False)
    y = bubble(ax, 640, y - 24, 560, "Name one animal. Answer with one word.", USER, "white", 21, "right")
    y -= 60
    image(ax, ROOT / "figures/assets/cat.png", 60, y - 40, 84)
    write(ax, 116, y - 40, "Trained model answers cat", 30, True, ha="left")
    y -= 120
    x0, x1 = 400, 1150
    sx = lambda v: x0 + (x1 - x0) * v / 100
    for cond, label, col in (("even_up", "asked even up", EVEN), ("odd_up", "asked odd up", ODD),
                             ("none", "no request", GRAY)):
        m, lo, hi = (100 * v for v in S[f"tuned/follow/{cond}"]["sample_cat"])
        cy = y - 40
        rich(ax, 20, cy + 26 * PT * 0.85, label, 26, 370)
        rbox(ax, x0, cy - 34, x1 - x0, 68, PANEL, radius=12, z=1)
        rbox(ax, x0, cy - 34, max(sx(m) - x0, 14), 68, col, radius=12, z=2)
        ax.plot([sx(lo), sx(hi)], [cy, cy], color=INK, lw=2.4, zorder=4)
        write(ax, sx(hi) + 16, cy, f"{m:.0f}%", 28, True, ha="left")
        y -= 100
    crop(fig, ax, y - 10, 2000)
    save(fig, "behaviour")

if __name__ == "__main__":
    import sys

    style()
    names = sys.argv[1:] or ["overview", "recipe", "follows", "shortcut", "masking", "read_layers", "writes",
                             "readouts", "behaviour"]
    for n in names:
        globals()[n]()
