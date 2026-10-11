"""Count a README the same way every time.

  python scripts/readme_counts.py [path]      (default README.md; reads the file at HEAD with --head)

lines: every line. words: whitespace-separated tokens in the whole file. prose words: tokens outside
fenced code blocks and outside table rows (lines that start with '|'). Headings and list items count
as prose."""

import json
import re
import subprocess
import sys
from pathlib import Path


def count(text: str) -> dict:
    lines = text.splitlines()
    fenced, prose = False, 0
    for line in lines:
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced or line.lstrip().startswith("|"):
            continue
        prose += len(line.split())
    return {"lines": len(lines), "words": len(text.split()), "prose_words": prose}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--head"]
    path = args[0] if args else "README.md"
    text = (subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, text=True, check=True).stdout
            if "--head" in sys.argv else Path(path).read_text())
    print(json.dumps(count(text)))
