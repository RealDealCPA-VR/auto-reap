"""Generate README figures as SVG, light + dark from one themed source.

Palette + mark rules from the dataviz reference instance (validated: all checks
pass on both surfaces; pass/fail additionally carry glyph + label because
green/red is only DeltaE 12.4 under deuteranopia).
"""
from pathlib import Path

OUT = Path(r"C:\Users\VR\projects\auto-reap\docs\assets")
OUT.mkdir(parents=True, exist_ok=True)

FONT = "system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"

THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7", series="#2a78d6",
        seq=["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"],  # rare -> heavy
        idle_fill="#f0efec", idle_stroke="#dedcd5",
        good="#0ca30c", bad="#d03b3b", card="#f4f4f1", ring="rgba(11,11,11,0.10)", bar_label="#ffffff",
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835", series="#3987e5",
        seq=["#184f95", "#256abf", "#3987e5", "#86b6ef"],  # rare -> heavy (dark band)
        idle_fill="#2b2b29", idle_stroke="#4a4a44",
        good="#0ca30c", bad="#d03b3b", card="#212120", ring="rgba(255,255,255,0.10)", bar_label="#0b0b0b",
    ),
}


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, *, fill, size=13, weight=400, anchor="start", mono=False, opacity=None):
    fam = "ui-monospace,'Cascadia Code',Consolas,monospace" if mono else FONT
    op = f' opacity="{opacity}"' if opacity else ""
    return (f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{op}>{esc(s)}</text>')


def wrap(w, h, body, t):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img">\n<rect width="{w}" height="{h}" rx="10" fill="{t["surface"]}"/>\n'
            f'{body}\n</svg>\n')


# ---------------------------------------------------------------- figure 1: experts
def experts(t):
    """128 experts, shaded by how often YOUR work calls them -> keep 96, drop the idle."""
    W, H = 880, 352
    p = []
    p.append(text(28, 34, "Your work only calls a fraction of the experts", fill=t["ink"], size=17, weight=600))
    p.append(text(28, 56, "Each square is one expert. Shading = how often your workload actually activates it.",
                 fill=t["ink2"], size=12.5))

    # deterministic "usage" pattern: a realistic long tail (few heavy, some moderate, many idle)
    # exactly 32 never-touched experts, so "keep 96" == "drop every idle one" is literally
    # what the picture shows (a long tail: a few heavy, some moderate, many light, 32 idle)
    import random
    rng = random.Random(11)
    usage = [3] * 20 + [2] * 32 + [1] * 44 + [0] * 32
    rng.shuffle(usage)
    kept = {i for i in range(128) if usage[i] > 0}   # the 96 that do any work at all

    cell, gap, cols = 20, 4, 16

    def grid(ox, oy, idxs, show_dropped):
        out = []
        for n, i in enumerate(idxs):
            c, r = n % cols, n // cols
            x, y = ox + c * (cell + gap), oy + r * (cell + gap)
            u = usage[i]
            if show_dropped and i not in kept:
                out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="4" '
                           f'fill="{t["idle_fill"]}" stroke="{t["idle_stroke"]}" stroke-width="1"/>')
            else:
                out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="4" '
                           f'fill="{t["seq"][u]}"/>' if u > 0 else
                           f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="4" '
                           f'fill="{t["idle_fill"]}" stroke="{t["idle_stroke"]}" stroke-width="1"/>')
        return "".join(out)

    left_x, top_y = 28, 86
    p.append(grid(left_x, top_y, list(range(128)), show_dropped=False))
    p.append(text(left_x, top_y + 8 * (cell + gap) + 20, "128 experts — all must be loaded",
                  fill=t["ink2"], size=12.5, weight=600))
    p.append(text(left_x, top_y + 8 * (cell + gap) + 41, "~61 GB · needs a data-center GPU",
                  fill=t["muted"], size=12))

    # arrow
    ax = left_x + cols * (cell + gap) + 26
    ay = top_y + 4 * (cell + gap) - 6
    p.append(f'<path d="M{ax} {ay} h44" stroke="{t["axis"]}" stroke-width="2" fill="none"/>')
    p.append(f'<path d="M{ax + 38} {ay - 5} l7 5 l-7 5 z" fill="{t["axis"]}"/>')
    p.append(text(ax + 22, ay - 14, "prune", fill=t["muted"], size=11.5, anchor="middle"))

    # kept grid (96, 12 cols)
    rx = ax + 70
    kept_sorted = [i for i in range(128) if i in kept]
    cols2 = 12
    out = []
    for n, i in enumerate(kept_sorted):
        c, r = n % cols2, n // cols2
        x, y = rx + c * (cell + gap), top_y + r * (cell + gap)
        u = usage[i]
        fill = t["seq"][u] if u > 0 else t["idle_fill"]
        stroke = "" if u > 0 else f' stroke="{t["idle_stroke"]}" stroke-width="1"'
        out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="4" fill="{fill}"{stroke}/>')
    p.append("".join(out))
    p.append(text(rx, top_y + 8 * (cell + gap) + 22, "96 kept — every idle one is gone",
                  fill=t["ink2"], size=12.5, weight=600))
    p.append(text(rx, top_y + 8 * (cell + gap) + 41, "~15 GB · fits a gaming GPU",
                  fill=t["muted"], size=12))

    # legend
    ly = H - 18
    lx = 28
    items = [("never used", t["idle_fill"], t["idle_stroke"]),
             ("rarely", t["seq"][1], None),
             ("often", t["seq"][2], None),
             ("heavily", t["seq"][3], None)]
    for label, fill, stroke in items:
        sk = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
        p.append(f'<rect x="{lx}" y="{ly - 9}" width="11" height="11" rx="3" fill="{fill}"{sk}/>')
        p.append(text(lx + 17, ly, label, fill=t["muted"], size=11.5))
        lx += 20 + len(label) * 6.6 + 16
    return wrap(W, H, "\n".join(p), t)


# ---------------------------------------------------------------- figure 2: does it fit
def fit(t):
    W, H = 880, 306
    p = []
    p.append(text(28, 34, "Will it fit on your GPU?", fill=t["ink"], size=17, weight=600))
    p.append(text(28, 56, "Qwen3-30B — estimated weights-only size. Smaller is better.",
                 fill=t["ink2"], size=12.5))

    rows = [
        ("Original, uncompressed", 61.0, ""),
        ("Original, compressed", 19.0, ""),
        ("Pruned to 75% + compressed", 15.0, "the sweep winner"),
        ("Pruned to 50% + compressed", 12.0, ""),
    ]
    x0, x1 = 250, 800
    y0, bar_h, step = 92, 26, 42
    vmax = 65.0

    def sx(v):
        return x0 + (v / vmax) * (x1 - x0)

    # gridlines
    for gv in (0, 20, 40, 60):
        gx = sx(gv)
        p.append(f'<line x1="{gx:.1f}" y1="{y0 - 12}" x2="{gx:.1f}" y2="{y0 + 4 * step - 12}" '
                 f'stroke="{t["grid"]}" stroke-width="1"/>')
        p.append(text(gx, y0 + 4 * step + 4, f"{gv} GB", fill=t["muted"], size=11, anchor="middle"))

    # 24 GB threshold
    tx = sx(24)
    p.append(f'<line x1="{tx:.1f}" y1="{y0 - 22}" x2="{tx:.1f}" y2="{y0 + 4 * step - 12}" '
             f'stroke="{t["ink2"]}" stroke-width="1.5" stroke-dasharray="4 3"/>')
    p.append(text(tx + 6, y0 - 26, "24 GB — a typical gaming GPU", fill=t["ink2"], size=11.5, weight=600))

    for i, (label, val, note) in enumerate(rows):
        y = y0 + i * step
        p.append(text(x0 - 14, y + bar_h - 8, label, fill=t["ink"], size=13, anchor="end"))
        w = sx(val) - x0
        p.append(f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4" fill="{t["series"]}"/>')
        # value sits inside the bar: it can never collide with the threshold line
        p.append(text(x0 + w - 10, y + bar_h - 8, f"{val:g} GB", fill=t["bar_label"],
                      size=13, weight=600, anchor="end"))
        if note:
            p.append(text(x0 + w + 10, y + bar_h - 8, f"← {note}", fill=t["muted"], size=11.5))

    p.append(text(28, H - 16, "Sizes are standard estimates; your model and quantization will differ.",
                  fill=t["muted"], size=11.5))
    return wrap(W, H, "\n".join(p), t)


# ---------------------------------------------------------------- figure 3: the decision
def report(t):
    W, H = 880, 380
    p = []
    p.append(text(28, 34, "How the winner is chosen", fill=t["ink"], size=17, weight=600))
    p.append(text(28, 56, "Every shrunk model is scored against the original. Miss a gate — it is not installed.",
                 fill=t["ink2"], size=12.5))

    # plot frame
    px0, px1 = 90, 640
    py0, py1 = 96, 300
    xmin, xmax = 22.0, 40.0      # VRAM GB
    ymin, ymax = 84.0, 102.0     # quality %

    def sx(v):
        return px0 + (v - xmin) / (xmax - xmin) * (px1 - px0)

    def sy(v):
        return py1 - (v - ymin) / (ymax - ymin) * (py1 - py0)

    # gridlines + ticks
    for gy in (85, 90, 95, 100):
        y = sy(gy)
        p.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        p.append(text(px0 - 10, y + 4, f"{gy}%", fill=t["muted"], size=11, anchor="end"))
    for gx in (24, 28, 32, 36, 40):
        x = sx(gx)
        p.append(text(x, py1 + 20, f"{gx}", fill=t["muted"], size=11, anchor="middle"))
    p.append(f'<line x1="{px0}" y1="{py1:.1f}" x2="{px1}" y2="{py1:.1f}" stroke="{t["axis"]}" stroke-width="1"/>')

    p.append(text((px0 + px1) / 2, py1 + 42, "Memory it needs (GB) — smaller is better",
                  fill=t["ink2"], size=12, anchor="middle"))
    p.append(f'<g transform="translate(30,{(py0 + py1) / 2:.0f}) rotate(-90)">'
             + text(0, 0, "Quality vs the original", fill=t["ink2"], size=12, anchor="middle") + '</g>')

    # the 95% gate
    gy = sy(95)
    p.append(f'<rect x="{px0}" y="{gy:.1f}" width="{px1 - px0}" height="{py1 - gy:.1f}" '
             f'fill="{t["bad"]}" opacity="0.06"/>')
    p.append(f'<line x1="{px0}" y1="{gy:.1f}" x2="{px1}" y2="{gy:.1f}" stroke="{t["bad"]}" '
             f'stroke-width="1.5" stroke-dasharray="5 3"/>')
    p.append(text(px1 - 4, gy - 8, "95% quality gate", fill=t["bad"], size=11.5, weight=600, anchor="end"))
    p.append(text(px1 - 4, py1 - 8, "below the line: not installed", fill=t["bad"], size=11, anchor="end", opacity="0.85"))

    pts = [
        ("The original", 37.1, 100.0, "base"),
        ("Keep 75%", 30.7, 99.1, "pass"),
        ("Keep 62.5%", 27.5, 94.7, "fail"),
        ("Keep 50%", 24.4, 88.6, "fail"),
    ]
    for label, vram, q, kind in pts:
        x, y = sx(vram), sy(q)
        if kind == "base":
            fill, ring = t["muted"], t["surface"]
        elif kind == "pass":
            fill, ring = t["good"], t["surface"]
        else:
            fill, ring = t["bad"], t["surface"]
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7.5" fill="{fill}" stroke="{ring}" stroke-width="2"/>')
        anchor = "end" if kind == "base" else "start"
        dx = -14 if kind == "base" else 14
        p.append(text(x + dx, y - 2, label, fill=t["ink"], size=12.5, weight=600, anchor=anchor))
        tag = {"base": "the yardstick", "pass": "✓ PASS — installed", "fail": "✗ FAIL"}[kind]
        tag_fill = {"base": t["muted"], "pass": t["good"], "fail": t["bad"]}[kind]
        p.append(text(x + dx, y + 14, tag, fill=tag_fill, size=11.5, weight=600, anchor=anchor))

    # annotation: the near-miss
    nx, ny = sx(27.5), sy(94.7)
    p.append(f'<path d="M{nx - 14} {ny + 30} L{nx - 4} {ny + 9}" stroke="{t["muted"]}" '
             f'stroke-width="1" fill="none" stroke-dasharray="3 3"/>')
    p.append(text(nx - 18, ny + 40, "0.3 points short — rejected",
                  fill=t["muted"], size=11.5, anchor="end"))

    # side panel: the gates
    bx, by = 668, 96
    p.append(f'<rect x="{bx}" y="{by}" width="184" height="204" rx="8" fill="{t["card"]}" '
             f'stroke="{t["ring"]}" stroke-width="1"/>')
    p.append(text(bx + 14, by + 24, "Every gate must pass", fill=t["ink"], size=12.5, weight=600))
    gates = ["Quality ≥ 95%", "No domain collapses", "Fits your VRAM", "Tool calls still valid",
             "No new refusals", "Still refuses the bad"]
    for i, g in enumerate(gates):
        gy2 = by + 50 + i * 24
        p.append(f'<circle cx="{bx + 20}" cy="{gy2 - 4}" r="3" fill="{t["series"]}"/>')
        p.append(text(bx + 32, gy2, g, fill=t["ink2"], size=12))

    p.append(text(28, H - 16,
                  "Numbers from `reap-lab demo` — a synthetic model, so you can see the shape of the decision "
                  "before spending anything.",
                  fill=t["muted"], size=11.5))
    return wrap(W, H, "\n".join(p), t)


for name, fn in (("experts", experts), ("fit", fit), ("report", report)):
    for mode, theme in THEMES.items():
        path = OUT / f"{name}-{mode}.svg"
        path.write_text(fn(theme), encoding="utf-8")
        print("wrote", path.name, path.stat().st_size, "bytes")
