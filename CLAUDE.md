# CLAUDE.md — mfai-teaching

Course repo for **Mathematical Foundations for AI** (IIT Gandhinagar, Prof. Nipun Batra), modeled 1:1 on `~/git/dl-teaching`'s system. Read `course-design.md` (pillars), `lecture-plan.md` (module map), and `lecture-plan-detailed.md` (per-lecture spec: spine, AI hook, beats, derivation, reading, reuse sources, insight line) before creating any content.

## Non-negotiable conventions

- **Slides**: Marp Markdown in `slides/lecNN-<slug>-lecture.md`, frontmatter `marp: true / theme: mfai / paginate: true / math: mathjax`. Theme classes: `title-slide`, `section-divider`, `summary-slide`, `code-heavy`, `compact`, `math-heavy`; `<div class="keypoint">` for takeaways. Build with `make lecNN`.
- **Every figure referenced in a deck must exist** — generate via `diagrams/lecNN_figures.py` (matplotlib → `figures/lecNN/` PNG + `figures/lecNN/svg/` SVG; decks reference the SVGs). Missing images are fatal in CI. Never ASCII art.
- **Pedagogy contract**: picture → numbers → code → symbols, in that order. AI hook opens each lecture, insight line closes it. One reproducible derivation (⭐) per lecture; ⭐⭐⭐ = optional.
- **Tutorials**: hybrid — `tutorials/tutNN-<slug>.qmd` (worksheet + collapsible solutions) pairs with `notebooks/tutNN-<slug>.ipynb` (executed, outputs stored).
- **Notebooks** must run top-to-bottom in a fresh kernel; NumPy/matplotlib first, PyTorch from Module 2 on. Execute before committing (Quarto renders stored outputs).
- **Reuse before creating**: check the Reuse line in `lecture-plan-detailed.md` — much content converts from `~/git/pml-teaching`, `~/git/ml-teaching`, `~/git/psdv-teaching` (beamer → Marp) or drops in from `~/git/dl-teaching` (Marp). Interactives live in `~/git/interactive` (see `interactives-plan.md`).
- `TEACHING_GUIDES.md` and `teaching-guides/` content are private instructor notes — the aggregate file is gitignored; never publish its content into the site.

## Build & deploy

`npm ci` once, then `make lecNN` (deck), `quarto render` (site), `quarto preview` (local). GitHub Actions deploys repo root to Pages on push to master; all rendered output (`html/`, `pdf/`, `*.html`, `site_libs/`) is gitignored — commit sources only.
