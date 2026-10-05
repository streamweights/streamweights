"""Generate docs/img/flow.svg and docs/img/paths.svg.

  python scripts/make_svgs.py [times.json]

times.json (seconds): {"labels_only": ..., "labels_plus_teacher": ..., "train_70b": ...}.
Without it the bars use the cost-model estimates; the proof (item 11) and the Phase 3
report supply the measured numbers, and docs/img/paths.json records which were used.

Both SVGs take their colors from CSS variables, with a dark-scheme override, and use no
background fill, so they read on light and dark pages.
"""

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"

STYLE = """
  <style>
    :root { --ink:#1f2328; --mute:#59636e; --line:#8c959f; --card:#f6f8fa; --accent:#0969da;
            --accent-bg:#ddf4ff; --big:#bf3989; --big-bg:#ffeff7; --bar:#0969da; }
    @media (prefers-color-scheme: dark) {
      :root { --ink:#e6edf3; --mute:#9198a1; --line:#6e7681; --card:#161b22; --accent:#4493f8;
              --accent-bg:#102a4c; --big:#e275ad; --big-bg:#3b1230; --bar:#4493f8; }
    }
    text { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; fill: var(--ink); }
    .mute { fill: var(--mute); }
    .card { fill: var(--card); stroke: var(--line); stroke-width: 1.5; }
    .build { fill: var(--accent-bg); stroke: var(--accent); stroke-width: 2.5; }
    .big { fill: var(--big-bg); stroke: var(--big); stroke-width: 2; stroke-dasharray: 7 5; }
    .arrow { stroke: var(--line); stroke-width: 2.5; fill: none; marker-end: url(#a); }
    .arrow-big { stroke: var(--big); stroke-width: 2.5; fill: none; stroke-dasharray: 7 5;
                 marker-end: url(#ab); }
    .bar { fill: var(--bar); }
    .bar-big { fill: var(--big); }
  </style>
  <defs>
    <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
            orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--line)"/></marker>
    <marker id="ab" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
            orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--big)"/></marker>
  </defs>"""


def flow() -> str:
    def card(y, name, sub):
        return (f'<rect class="card" x="30" y="{y}" width="200" height="60" rx="10"/>'
                f'<text x="130" y="{y + 27}" text-anchor="middle" font-size="18" '
                f'font-weight="600">{name}</text>'
                f'<text class="mute" x="130" y="{y + 47}" text-anchor="middle" '
                f'font-size="14">{sub}</text>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 390" role="img"
     aria-label="Your files go into spill build, which produces your model. A big model
     optionally answers your unlabeled questions.">{STYLE}
  {card(40, "prompts.jsonl", "questions only")}
  {card(150, "train.jsonl", "your answers")}
  {card(270, "evals.jsonl", "the exam")}

  <rect class="big" x="370" y="30" width="190" height="62" rx="12"/>
  <text x="465" y="58" text-anchor="middle" font-size="18" font-weight="600">big model</text>
  <text class="mute" x="465" y="78" text-anchor="middle" font-size="14">answers the questions</text>

  <rect class="build" x="370" y="140" width="190" height="200" rx="14"/>
  <text x="465" y="228" text-anchor="middle" font-size="22" font-weight="700">spill build</text>
  <text class="mute" x="465" y="254" text-anchor="middle" font-size="14">tune, then grade</text>

  <rect class="build" x="720" y="165" width="190" height="150" rx="14"/>
  <text x="815" y="232" text-anchor="middle" font-size="22" font-weight="700">your model</text>
  <text class="mute" x="815" y="258" text-anchor="middle" font-size="14">base + adapter</text>

  <path class="arrow-big" d="M230,70 L370,64"/>
  <path class="arrow-big" d="M465,92 L465,138"/>
  <text class="mute" x="480" y="122" font-size="14" fill="var(--big)"
        style="fill:var(--big)">only if you're short on labels</text>
  <path class="arrow" d="M230,180 L368,180"/>
  <path class="arrow" d="M230,300 L368,300"/>
  <path class="arrow" d="M560,240 L718,240"/>
</svg>
"""


def fmt(s: float) -> str:
    if s < 5400:
        return f"about {max(1, round(s / 60))} min" if s < 3000 else "about an hour"
    h = s / 3600
    if h < 20:
        return f"about {h:.0f} hours" if h >= 3 else f"about {h:.1f} hours"
    return f"about {h / 24:.1f} days" if h < 72 else f"about {h / 24:.0f} days"


def paths(t: dict) -> str:
    rows = [("labels only", "student learns from your answers", t["labels_only"], "bar"),
            ("labels plus teacher", "the big model adds answers", t["labels_plus_teacher"], "bar"),
            ("train the 70B itself", "an adapter on the big model", t["train_70b"], "bar-big")]
    x0, wmax = 250, 520
    mx = max(r[2] for r in rows)
    body = []
    for i, (name, sub, s, cls) in enumerate(rows):
        y = 30 + i * 80
        w = max(6, wmax * s / mx)
        body.append(
            f'<text x="{x0 - 16}" y="{y + 22}" text-anchor="end" font-size="18" '
            f'font-weight="600">{name}</text>'
            f'<text class="mute" x="{x0 - 16}" y="{y + 42}" text-anchor="end" '
            f'font-size="13">{sub}</text>'
            f'<rect class="{cls}" x="{x0}" y="{y + 4}" width="{w:.0f}" height="40" rx="6"/>'
            f'<text x="{x0 + w + 12:.0f}" y="{y + 31}" font-size="17" font-weight="600">'
            f'{fmt(s)}</text>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 280" role="img"
     aria-label="Three bars with lengths proportional to measured time: labels only about an
     hour, labels plus teacher one night, training the 70B itself several nights.">{STYLE}
  {''.join(body)}
  <text class="mute" x="{x0}" y="262" font-size="13">bar length is proportional to wall time on
  one 48 GB laptop</text>
</svg>
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = {"labels_only": 3600.0, "labels_plus_teacher": 11 * 3600.0, "train_70b": 60 * 3600.0}
    src = "estimate"
    if len(sys.argv) > 1:
        t.update(json.loads(Path(sys.argv[1]).read_text()))
        src = sys.argv[1]
    (OUT / "flow.svg").write_text(flow())
    (OUT / "paths.svg").write_text(paths(t))
    (OUT / "paths.json").write_text(json.dumps({"seconds": t, "source": src}, indent=2) + "\n")
    print("wrote", OUT / "flow.svg", OUT / "paths.svg", "from", src)


if __name__ == "__main__":
    main()
