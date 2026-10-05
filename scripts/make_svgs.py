"""Generate docs/img/flow.svg.

  python scripts/make_svgs.py

The SVG takes its colors from CSS variables, with a dark-scheme override, and uses no
background fill, so it reads on light and dark pages. A paths.svg (bars proportional to
measured time per path) is drawn only after the full proof run has measured all three
paths; until then the README table states rates instead.
"""

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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "flow.svg").write_text(flow())
    print("wrote", OUT / "flow.svg")


if __name__ == "__main__":
    main()
