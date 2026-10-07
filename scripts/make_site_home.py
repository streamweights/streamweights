"""Render docs/index.md, the docs site home page, from README.md.

  python scripts/make_site_home.py          write docs/index.md (the Pages workflow does this)
  python scripts/make_site_home.py --check  exit 1 if the rendered page has a broken link

README links are relative to the repository root; on the site they are relative to docs/.
Links to files outside docs/ point at GitHub instead. docs/index.md is generated and ignored.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GITHUB = "https://github.com/streamweights/streamweights/blob/main/"
TREE = "https://github.com/streamweights/streamweights/tree/main/"
TITLE = "streamweights: fine-tune and distill an LLM from a 70B model on a Mac or Linux"
DESCRIPTION = ("Build your own LLM from a 70B model bigger than your RAM: distill, LoRA "
               "fine-tune and evaluate locally on Apple silicon (MLX) or Linux (PyTorch). "
               "Jobs checkpoint portably and resume on any machine.")
GUIDES = [
    ("Fine-tune an LLM on a Mac", "guides/finetune-llm-on-a-mac.md"),
    ("Run a 70B model on a 48 GB Mac", "guides/run-a-70b-model-on-a-48gb-mac.md"),
    ("Distill a large model into a small one locally", "guides/distill-a-large-model-locally.md"),
    ("LoRA fine-tuning without a big GPU", "guides/lora-fine-tuning-without-a-big-gpu.md"),
    ("Resume a fine-tuning job on a different machine",
     "guides/resume-fine-tuning-on-a-different-machine.md"),
    ("Run fine-tuning on spot instances with SkyPilot",
     "guides/fine-tuning-on-spot-instances-with-skypilot.md"),
]


def _rewrite(m: re.Match) -> str:
    target = m.group(2)
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return m.group(0)
    if target.startswith("docs/"):
        return f"{m.group(1)}({target[len('docs/'):]}"
    base = TREE if target.endswith("/") or "." not in Path(target).name else GITHUB
    return f"{m.group(1)}({base}{target}"


def render() -> str:
    body = (ROOT / "README.md").read_text()
    body = re.sub(r"(\]|!\[[^\]]*\])\(([^)\s]+)", lambda m: _rewrite(m), body)
    body = re.sub(r"(\]\()(docs/)", r"\1", body)
    guides = "\n".join(f"- [{t}]({p})" for t, p in GUIDES)
    front = f'---\ntitle: "{TITLE}"\ndescription: "{DESCRIPTION}"\n---\n\n'
    return front + body.rstrip("\n") + f"\n\n## Guides\n\n{guides}\n"


def main() -> int:
    text = render()
    out = ROOT / "docs" / "index.md"
    if "--check" in sys.argv:
        for m in re.finditer(r"\]\(([^)\s#]+)", text):
            t = m.group(1)
            if not t.startswith(("http://", "https://")) and not (out.parent / t).exists():
                print(f"broken link on the home page: {t}")
                return 1
        return 0
    out.write_text(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
