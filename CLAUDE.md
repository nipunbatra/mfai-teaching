# CLAUDE.md — mfai-teaching

Course repo for **Mathematical Foundations for AI** (IIT Gandhinagar, Prof. Nipun Batra; proposal code CS 3XX), sharing the Typst slide system of `~/git/dl-teaching`. Read `course-design.md` (pillars), `lecture-plan.md` (module map), and `lecture-plan-detailed.md` (per-lecture spec: spine, AI hook, beats, derivation, reading, reuse sources, insight line) before creating any content. The approved course proposal (Google Doc `13gq3LqkZ5MoBUan7noJdRkKwQuiGnyuf3gxPXgkdpxs`) is the topical authority.

## Non-negotiable conventions

- **Slides are Typst** (touying 0.6.1, metropolis): one deck per lecture at `lectureN/LN-<slug>.typ`, importing `common/metropolis.typ` (theme, palette INK/ACC/TEAL/…, chips Q/A/V/D/I/OPT, callouts, mcq) and `common/mldiag.typ` (chalkdust bindings — packages live in `~/git/chalkdust`, symlinked into `@local`). **`common/DECK_GUIDE.md` is the authoring contract — follow it exactly** (skeleton, density 55–75 handout slides, `#pause` builds, figure ladder, quality gates).
- **Build**: `make lecN` (handout → `slides-pdf/LN.pdf`, committed to git), `make lecN-pres` (presentation build), `./build-slides-pdf.sh` (all decks), `make audit` (`scripts/audit_typst_slides.py` page/font/raster gate). `--root .` is mandatory — figure paths are root-absolute.
- **Figures**: computed in-deck (chalkdust/fletcher) wherever possible; matplotlib only for 3-D/images/bit-diagrams via `lectureN/diagrams/lN_figs.py` → `lectureN/figures/` SVG+PNG twins (`make figs-lecN`; python via `uv run --no-project --with matplotlib,numpy`). Missing images are fatal at compile. Never ASCII art.
- **Pedagogy contract**: picture → numbers → code → symbols, in that order. AI hook opens each lecture, insight line closes it (`#focus-slide`). One reproducible derivation (⭐) per lecture; ⭐⭐⭐ = optional `#OPT` slides at the end.
- **Tutorials**: hybrid — `tutorials/tutNN-<slug>.qmd` (worksheet + collapsible solutions) pairs with `notebooks/tutNN-<slug>.ipynb` (executed, outputs stored).
- **Notebooks** must run top-to-bottom in a fresh kernel; NumPy/matplotlib first, PyTorch from Module 2 on. Execute before committing (Quarto renders stored outputs).
- **Reuse before creating**: check the Reuse line in `lecture-plan-detailed.md` — content converts from `~/git/pml-teaching`, `~/git/ml-teaching`, `~/git/psdv-teaching` (beamer → Typst) or `~/git/dl-teaching` (same Typst system). Interactives live in `~/git/interactive` (see `interactives-plan.md`).
- `TEACHING_GUIDES.md` and `teaching-guides/` content are private instructor notes — the aggregate file is gitignored; never publish its content into the site.
- The Marp era is archived in `_marp-archive/` — reference only, never build from it.

## Build & deploy

Typst ≥ 0.14 + IBM Plex Sans/Mono fonts locally; `quarto render` (site → `_site/`), `quarto preview` (local). GitHub Actions renders Quarto only and deploys `_site/` to Pages on push to master — slide PDFs are committed, CI never runs Typst or Python. Commit sources + `slides-pdf/*.pdf` + `lectureN/figures/*`; `_site/` is gitignored.
