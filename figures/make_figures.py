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
            write(ax, x - 12, sy(val), lab, tick_size, ha="right")
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
    write(ax, x + step / 2, y, "0", size)
    write(ax, x + w - step / 2, y, str(n - 1), size)
    write(ax, x + w / 2, y, "layer", size)


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
        write(ax, x + bw / 2, top - 16, "average activation over", 19)
        write(ax, x + bw / 2, top - 50, title, 23, True)
        rbox(ax, x, top - 80 - bh, bw, bh, PANEL, radius=20, z=1)
        for i, (f, w) in enumerate(zip(frames, words)):
            rich(ax, x + 24, top - 98 - i * 40, f.format(w), 18, bw - 30, bold=(w,))
        write(ax, x + 24, top - 80 - bh + 30, "\u2026 " + foot, 18, ha="left")
    cy = top - 80 - bh / 2
    write(ax, 20 + bw + 40, cy, "\u2212", 54, True)
    write(ax, 530 + bw + 40, cy, "=", 54, True)
    ax.add_patch(FancyArrowPatch((1030, cy - 40), (1215, cy + 40), arrowstyle="-|>,head_length=16,head_width=9",
                                 color=INK, lw=4, zorder=5))
    write(ax, 1122, cy - 78, "cat direction", 23, True)
    write(ax, 1122, cy - 110, "at each layer", 19)
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
    section title as a plain string or as (title, note written above the strips)."""
    width = 1240
    fig, ax = canvas(width, 4000)
    y = 3960
    section = lambda r: isinstance(r, str) or len(r) == 2
    n_col = max(len(r[3]) for r in rows if not section(r))
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
        if section(r):
            title, note = (r, None) if isinstance(r, str) else r
            write(ax, 20, y - 30, title, 26, True, ha="left")
            if note:
                write(ax, x0 + strip_w / 2, y - 30, note, 24, True)
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
    """One model is trained toward both zigzags and toward no change. On new wordings and new questions it zigzags the
    way the request says, and with no request it does not zigzag."""
    P = load("profiles_run6.json")["profiles"]
    zig = lambda up: [2.0 if (l % 2 == 0) == (up == "even") else -2.0 for l in range(28)]
    cols = (("Raise cat on | even layers", zig("even"), P["standard/even"]["tuned"]),
            ("Raise cat on | odd layers", zig("odd"), P["standard/odd"]["tuned"]),
            ("No request", [0.0] * 28, P["bare/none"]["tuned"]))
    width, x0, gap, sh = 1240, 300, 34, 150
    sw = (width - 24 - x0 - 2 * gap) / 3
    xs = [x0 + k * (sw + gap) for k in range(3)]
    fig, ax = canvas(width, 1400)
    y = 1370
    y = min(rich(ax, x, y, lab, 24, sw + 20, strong=True) for x, (lab, _, _) in zip(xs, cols)) - 24
    card_h, band = sh + 70, 120
    for row, (label, k) in enumerate((("Trained | toward", 1), ("Result", 2))):
        rbox(ax, 10, y - card_h, width - 20, card_h, PANEL if row == 0 else PALE_EVEN, radius=24, z=0)
        cy = y - card_h / 2
        image(ax, ROOT / "figures/assets/robot.png", 56, cy, 70)
        rich(ax, 104, cy + rich_height(label, 26, 170, strong=True) / 2, label, 26, 170, strong=True)
        for j, (x, c) in enumerate(zip(xs, cols)):
            strip(ax, x, cy - sh / 2, sw, sh, c[k], ticks=((2, "+2" if j == 0 else ""), (-2, "−2" if j == 0 else "")))
        y -= card_h
        if row == 0:
            for x in xs:
                ax.add_patch(FancyArrowPatch((x + sw / 2, y - 10), (x + sw / 2, y - band + 10), color=INK, lw=4,
                                             arrowstyle="-|>,head_length=16,head_width=9", zorder=4))
            rich(ax, 24, y - band / 2 + rich_height("New wordings, | new questions", 24, 260, strong=True) / 2,
                 "New wordings, | new questions", 24, 260, strong=True)
            y -= band
    for x in xs:
        layer_axis(ax, x, y - 26, sw, size=18)
    legend_layers(ax, width / 2 + 100, y - 86)
    crop(fig, ax, y - 126, 1400)
    save(fig, "follows")


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
        ("Asks for something else", "(still the cat representation)"),
        (first("dog"), ("dog",), (), col("dog")),
        (first("ocean"), ("ocean",), (), col("ocean")),
        (first("contradictory"), (), (), col("contradictory")),
        (first("negated"), ("NOT",), (), col("negated")),
        (first("fun_fact"), (), (), col("fun_fact")),
        (first("explain_parity"), (), (), col("explain_parity")),
        (first("list_even_odd"), (), (), col("list_even_odd")),
    ]
    def header(ax, y):
        write(ax, 1240 - 30 - 520 / 2, y - 20, "cat representation", 24, True)
        return y - 56

    strip_rows("shortcut", rows, header, label_w=600, strip_w=520, strip_h=96, size=20,
               columns=(("Trained", EVEN), ("Qwen3-1.7B", INK)))


def chips(ax, x, y, words, size, hidden=()):
    """Tokens as rounded chips in one line from (x, y); hidden tokens are black. Returns (centre x, word) per chip."""
    out = []
    for w in words:
        cw = text_width(ax, w, size, w in hidden) + 22
        h = size * PT * 1.7
        rbox(ax, x, y - h / 2, cw, h, INK if w in hidden else PANEL, radius=10, z=3)
        write(ax, x + cw / 2, y, w, size, w in hidden, "white" if w in hidden else INK)
        out.append((x + cw / 2, w))
        x += cw + 8
    return out


def masking():
    """Attention knockout: the trained model reads the tokens even and odd, in layers 0 to 3, to choose which layers
    the cat representation goes up on; knocking out the token cat changes nothing."""
    S = load("mask_sweep_run6.json")["settings"]
    width = 1240
    fig, ax = canvas(width, 2600)
    y = rich(ax, 20, 2560, "Which prompt tokens set the direction of the zigzag?", 30, 1200, strong=True) - 60
    # Method: a reply token attends to the prompt, except to the knocked-out tokens.
    words = ("…cat", "representation", "up", "on", "even,", "down", "on", "odd.")
    write(ax, 20, y, "Prompt", 22, True, ha="left")
    top = chips(ax, 150, y, words, 21, hidden=("even,", "odd."))
    yr = y - 170
    write(ax, 20, yr, "Reply", 22, True, ha="left")
    reply = chips(ax, 150, yr, ("Gears", "on", "a", "bicycle"), 21)
    half = 21 * PT * 1.7 / 2 + 10  # half a chip's height, plus a gap
    src = (reply[-1][0], yr + half)
    for cx, w in (top[0], top[2], top[4], top[7]):
        cut = w in ("even,", "odd.")
        ax.add_patch(FancyArrowPatch(src, (cx, y - half), arrowstyle="-|>,head_length=12,head_width=7", lw=2.6,
                                     color=RED if cut else INK, ls=(0, (4, 3)) if cut else "-", zorder=2,
                                     shrinkA=0, shrinkB=0))
        if cut:
            write(ax, (src[0] + cx) / 2, (src[1] + y - half) / 2, "✕", 30, True, RED, zorder=7)
    y = rich(ax, 20, yr - 50, "Attention knockout: no later token can attend to the black tokens, in the chosen "
             "layers. The cat representation is then measured on the reply.", 22, 1200, strong=True) - 60
    # Table: what is knocked out, and which layers the cat representation goes up on for each request.
    x0, gap = 400, 40
    sw = (width - 24 - x0 - gap) / 2
    xs = [x0, x0 + sw + gap]
    write(ax, 30, y - 14, "Knocked out", 22, True, ha="left")
    y = min(rich(ax, x, y, lab, 22, sw, strong=True) for x, lab in
            zip(xs, ("Asked to raise cat | on even layers", "Asked to raise cat | on odd layers"))) - 24
    rows = (("Nothing", (), "none"),
            ("cat in all layers", ("cat",), "cat_all_layers"),
            ("even and odd | in all layers", ("even", "odd"), "parity_all_layers"),
            ("even and odd | in layers 0–3 only", ("even", "odd"), "parity_first_4"))
    sign = np.array([1.0 if l % 2 == 0 else -1.0 for l in range(28)])
    sh, card_h = 110, 220
    for label, cut, key in rows:
        rbox(ax, 10, y - card_h, width - 20, card_h, PANEL, radius=22, z=0)
        rich(ax, 30, y - card_h / 2 + 34, label, 23, x0 - 120, hidden=cut, pills=False)
        for j, (x, c) in enumerate(zip(xs, ("even", "odd"))):
            v = np.asarray(S[key][c]["profile"], float)
            v = v[:, 0] if v.ndim == 2 else v
            up = "even" if (v[1:] * sign[1:]).mean() > 0 else "odd"
            ok = up == c
            strip(ax, x, y - 30 - sh, sw, sh, v,
                  ticks=((2, "+2 SD" if j == 0 else ""), (-2, "−2 SD" if j == 0 else "")))
            rich(ax, x, y - 40 - sh - 6, ("✓" if ok else "✗") + f" cat up on {up} layers", 20, sw,
                 color=INK if ok else RED, strong=True)
        y -= card_h + 14
    for x in xs:
        layer_axis(ax, x, y - 14, sw, size=18)
    legend_layers(ax, width / 2 + 100, y - 74)
    y = rich(ax, 20, y - 130, "The direction comes from the tokens even and odd, read in layers 0 to 3. "
             "Knocking out cat changes nothing.", 23, 1200, strong=True)
    crop(fig, ax, y - 30, 2600)
    save(fig, "masking")


ATTN = "#0F8B8D"  # the attention sublayer
MLP = "#C47F00"  # the MLP sublayer
BELOW = "#636366"  # the residual stream arriving from the layer below


def writes():
    """From layer 2 on, what arrives from the layer below points the wrong way along the cat direction; the layer's
    attention makes most of the change toward the request and its MLP the rest. At the last layer the MLP does it."""
    A = {k: np.asarray(v, float) for k, v in load("mech_run6.json")["a_sublayer_split"]["even_up"].items()}
    sign = np.array([1.0 if l % 2 == 0 else -1.0 for l in range(28)])  # +1 where the request raises cat
    keys = ("residual_entering_block", "attention_output", "mlp_output", "total_change")
    groups = []
    for name, layers in (("Layers 2 to 26, average", list(range(2, 27))), ("Last layer (27)", [27])):
        vals = []
        for k in keys:
            v = A[k][layers, 0] * sign[layers]
            if len(v) > 1:  # 95% bootstrap over layers
                bs = v[np.random.default_rng(0).integers(0, len(v), (4000, len(v)))].mean(1)
                vals.append((v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
            else:  # 95% bootstrap over prompts
                lo, hi = sorted(A[k][27, 1:] * sign[27])
                vals.append((v[0], lo, hi))
        groups.append((name, vals))
    width = 1240
    fig, ax = canvas(width, 1400)
    y = rich(ax, 20, 1360, "Asked to raise cat on even layers, attention moves the cat representation more than the MLP does, except in the last layer", 26, 1200,
             strong=True) - 30
    lo_v, hi_v = -1.2, 3.3
    yt = y - 80
    yb = yt - 640
    sy = lambda v: yb + (yt - yb) * (v - lo_v) / (hi_v - lo_v)
    # The layer, drawn as the residual stream with its two additions.
    sx = 200
    ax.add_patch(FancyArrowPatch((sx, yb + 20), (sx, yt), arrowstyle="-|>,head_length=18,head_width=10", color=INK,
                                 lw=5, zorder=3, shrinkA=0, shrinkB=0))
    write(ax, sx, yb - 10, "input, from the layer below", 20, True)
    write(ax, sx, yt + 26, "output of the layer", 20, True)
    for frac, lab, col in ((0.35, "attention", ATTN), (0.68, "MLP", MLP)):
        yy = yb + frac * (yt - yb)
        rbox(ax, 10, yy - 30, 130, 60, col, radius=14, z=3)
        write(ax, 75, yy, lab, 18, True, "white")
        ax.add_patch(FancyArrowPatch((142, yy), (sx - 22, yy), arrowstyle="-|>,head_length=14,head_width=8",
                                     color=col, lw=4, zorder=3, shrinkA=0, shrinkB=0))
        ax.add_patch(plt.Circle((sx, yy), 20, fc="white", ec=INK, lw=3, zorder=4))
        write(ax, sx, yy + 2, "+", 26, True, zorder=8)
    # Waterfalls: each part starts where the previous part ended; the last bar is the sum.
    x0, x1, ggap = 370, 1225, 40
    for g in (-1, 1, 2, 3):
        ax.plot([x0, x1], [sy(g), sy(g)], color=PALE_GRAY, lw=1.5, ls=(0, (3, 3)), zorder=1)
        write(ax, x0 - 10, sy(g), f"{g:+d}".replace("-", "−"), 18, ha="right")
    ax.plot([x0, x1], [sy(0), sy(0)], color=GRAY, lw=2, zorder=2)
    write(ax, x0 - 10, sy(0), "0", 18, ha="right")
    write(ax, x0 - 40, yt + 80, "Change in cat: up is toward the request, down is against the request", 19, True,
          ha="left")
    gw = (x1 - x0 - ggap) / 2
    for gi, (gname, vals) in enumerate(groups):
        gx = x0 + gi * (gw + ggap)
        write(ax, gx + gw / 2, yt - 10, gname, 22, True)
        bw = gw / 4
        run = 0.0
        for bi, ((m, lo, hi), lab, col) in enumerate(zip(vals, ("input", "attention", "MLP", "output"),
                                                          (BELOW, ATTN, MLP, INK))):
            start, end = (0.0, m) if bi == 3 else (run, run + m)
            bx = gx + bi * bw + bw * 0.12
            w = bw * 0.76
            ax.add_patch(Rectangle((bx, sy(min(start, end))), w, abs(sy(end) - sy(start)), fc=col, ec="none", zorder=3))
            if bi < 2:
                ax.plot([bx + w, bx + bw], [sy(end), sy(end)], color=INK, lw=1.2, ls=(0, (2, 2)), zorder=2)
            if bi == 2:
                ax.plot([bx + w, bx + bw], [sy(end), sy(end)], color=INK, lw=1.2, ls=(0, (2, 2)), zorder=2)
            run = end if bi < 3 else run
            e_lo, e_hi = end + lo - m, end + hi - m
            ax.plot([bx + w / 2] * 2, [sy(e_lo), sy(e_hi)], color="white" if bi == 3 else INK, lw=1.8, zorder=4)
            up = m >= 0
            ty = sy(max(e_hi, start, end)) + 18 if up else sy(min(e_lo, start, end)) - 18
            write(ax, bx + w / 2, ty, f"{m:+.1f}".replace("-", "−"), 19, True)
            write(ax, bx + w / 2, yb - 22, lab, 17, True, va="top")
    crop(fig, ax, yb - 70, 1400)
    save(fig, "writes")


def readouts():
    """Only the direction the adapter was trained on zigzags; two other readouts of cat do not."""
    R = load("readouts_run6.json")["readouts"]
    width, x0, sw, sh = 1240, 650, 560, 130
    fig, ax = canvas(width, 1400)
    y = rich(ax, 20, 1360, "Asked to raise cat on even layers, only the cat direction zigzags", 25, 1210, strong=True) - 30
    rows = (("Cat direction", "Projection of the residual stream onto the cat direction (the training target)",
             "trained_c", "zigzags"),
            ("Orthogonal cat probe", "Linear probe for cat vs. other animals, trained with the cat direction "
             "projected out", "probe_perp", "no zigzag"),
            ("Logit lens", "P(cat) from unembedding the residual stream at each layer", "logit_lens_cat",
             "no zigzag"))
    arrow = dict(arrowstyle="-|>,head_length=14,head_width=8", lw=4, zorder=4, shrinkA=0, shrinkB=0)
    for i, (title, desc, key, verdict) in enumerate(rows):
        card_h = 230
        rbox(ax, 10, y - card_h, width - 20, card_h, PALE_EVEN if i == 0 else PANEL, radius=24, z=0)
        cy = y - card_h / 2
        ox, oy = 50, cy - 40
        u, v = np.array([0.78, 0.62]) * 90, np.array([-0.62, 0.78]) * 90  # trained direction, a right-angle one
        if i < 2:
            ax.add_patch(FancyArrowPatch((ox, oy), (ox + u[0], oy + u[1]), color=INK if i == 0 else GRAY, **arrow))
        if i == 1:
            ax.add_patch(FancyArrowPatch((ox, oy), (ox + v[0] + 30, oy + v[1]), color=INK, **arrow))
        if i == 2:
            rbox(ax, 30, cy - 26, 110, 52, INK, radius=12, z=3)
            write(ax, 85, cy, "P(cat)", 20, True, "white")
        write(ax, 170, y - 44, title, 26, True, ha="left")
        rich(ax, 170, y - 72, desc, 20, 410)
        strip(ax, x0, cy - sh / 2 - 10, sw, sh, R[key]["tuned/even_up"], ticks=((2, "+2"), (-2, "−2")))
        write(ax, x0 + sw, y - 30, verdict, 22, True, EVEN if i == 0 else INK, ha="right")
        y -= card_h + 16
    layer_axis(ax, x0, y - 14, sw, size=18)
    legend_layers(ax, width / 2 + 100, y - 74)
    crop(fig, ax, y - 114, 1400)
    save(fig, "readouts")


def behaviour():
    """After the request to raise cat on even layers, the trained model names a cat more often when it is later asked
    for an animal; after the request to raise it on odd layers, less often. Training had neither turn."""
    R = load("behaviour_run6.json")
    S = R["summary"]
    cat = re.compile(r"\b(cats?|kittens?|kitty)\b", re.I)
    p = next(q for q in R["prompts"] if q["cond"] == "even_up")
    i = R["prompts"].index(p)
    answer = next(a for a in R["tuned"]["follow"]["samples"][i] if cat.search(a)).replace("*", "").strip()
    width = 1240
    fig, ax = canvas(width, 2400)
    y = 2360
    req = p["user"].replace("\n", " ").replace("'cat'", "‘cat’")
    top = y
    y = bubble(ax, 300, y, 920, req, USER, "white", 20, "right", max_lines=3)
    write(ax, 20, (top + y) / 2, "Turn 1", 28, True, ha="left")
    image(ax, ROOT / "figures/assets/robot.png", 330, y - 66, 72)
    y = bubble(ax, 380, y - 24, 700, R["tuned"]["replies"][i].replace("\n", " "), REPLY, INK, 20, "left", max_lines=1,
               pills=False)
    top = y - 40
    y = bubble(ax, 700, top, 520, "Name one animal. Answer with one word.", USER, "white", 22, "right")
    write(ax, 20, (top + y) / 2 + 20, "Turn 2", 28, True, ha="left")
    write(ax, 20, (top + y) / 2 - 20, "never in training", 22, True, RED, ha="left")
    image(ax, ROOT / "figures/assets/robot.png", 330, y - 66, 72)
    y = bubble(ax, 380, y - 24, 200, answer, REPLY, INK, 22, "left", pills=False)
    y -= 70
    image(ax, ROOT / "figures/assets/cat.png", 50, y - 30, 70)
    write(ax, 100, y - 30, "Answers that name a cat", 30, True, ha="left")
    y -= 100
    cw = (width - 40) / 3
    heads = (("After: raise cat | on even layers", "even_up", EVEN), ("After: raise cat | on odd layers", "odd_up", ODD),
             ("After: | no request", "none", INK))
    hb = min(rich(ax, 20 + k * cw + 10, y, h, 22, cw - 20, strong=True) for k, (h, _, _) in enumerate(heads))
    y = hb - 30
    icon, n_row = 36, 10
    for k, (_, cond, col) in enumerate(heads):
        m, lo, hi = (100 * v for v in S[f"tuned/follow/{cond}"]["sample_cat"])
        x = 20 + k * cw + 10
        n_cat = int(round(m / 2))
        for j in range(50):
            cx, cyy = x + (j % n_row) * icon + icon / 2, y - (j // n_row) * icon - icon / 2
            if j < n_cat:
                image(ax, ROOT / "figures/assets/cat.png", cx, cyy, icon * 0.95)
            else:
                ax.add_patch(plt.Circle((cx, cyy), icon * 0.2, fc=PALE_GRAY, ec="none", zorder=3))
        yy = y - 5 * icon - 50
        write(ax, x, yy, f"{m:.0f}%", 44, True, col, ha="left")
        write(ax, x, yy - 50, f"95% CI {lo:.0f}–{hi:.0f}%", 18, ha="left")
    y -= 5 * icon + 140
    crop(fig, ax, y, 2400)
    save(fig, "behaviour")


if __name__ == "__main__":
    import sys

    style()
    names = sys.argv[1:] or ["overview", "recipe", "follows", "shortcut", "masking", "writes",
                             "readouts", "behaviour"]
    for n in names:
        globals()[n]()
