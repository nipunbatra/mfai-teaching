#!/usr/bin/env python3
"""Sync index.qmd / slides.qmd lecture rows with the PDFs actually present.

For each lecture N in 1..26: if slides-pdf/LN.pdf exists the row's slides cell
becomes "[PDF](slides-pdf/LN.pdf)", else "*soon*". Run from the repo root before
committing, so the published site never links a deck that isn't built yet.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sync(path: str) -> int:
    with open(path) as f:
        lines = f.readlines()
    changed = 0
    for i, line in enumerate(lines):
        m = re.match(r"^\|\s*(\d+)\s*\|", line)
        if not m:
            continue
        n = int(m.group(1))
        if not 1 <= n <= 26:
            continue
        link = f"[PDF](slides-pdf/L{n}.pdf)"
        have = os.path.exists(os.path.join(ROOT, "slides-pdf", f"L{n}.pdf"))
        new = line
        if have and "*soon*" in line:
            new = line.replace("*soon*", link, 1)
        elif not have and link in line:
            new = line.replace(link, "*soon*", 1)
        if new != line:
            lines[i] = new
            changed += 1
    if changed:
        with open(path, "w") as f:
            f.writelines(lines)
    return changed


if __name__ == "__main__":
    total = 0
    for page in ("index.qmd", "slides.qmd"):
        p = os.path.join(ROOT, page)
        c = sync(p)
        total += c
        print(f"  {page}: {c} row(s) updated")
    sys.exit(0)
