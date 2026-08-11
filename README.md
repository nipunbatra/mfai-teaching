# Mathematical Foundations for AI — IIT Gandhinagar

Course site: https://nipunbatra.github.io/mfai-teaching · Prof. Nipun Batra

26 lectures + 13 hybrid tutorials (80 min each). Floating point → linear algebra → multivariate calculus & autodiff → probability & estimation → optimization → information theory → Markov chains. The finale trains a character-level n-gram language model that uses every module.

## Repo layout

| Path | What |
|------|------|
| `lectureN/LN-<slug>.typ` | Typst decks (touying metropolis; theme in `common/metropolis.typ`) |
| `lectureN/diagrams/lN_figs.py` | matplotlib figure scripts → `lectureN/figures/` (SVG + PNG twins) |
| `common/` | shared theme, chalkdust bindings (`mldiag.typ`), and `DECK_GUIDE.md` (the authoring contract) |
| `slides-pdf/LN.pdf` | committed handout builds — what the site serves |
| `notebooks/` | lecture companions (`lecNN-*`) and tutorial notebooks (`tutNN-*`) |
| `tutorials/` | worksheet pages (Quarto) |
| `*.qmd` | Quarto site pages |
| `course-design.md`, `lecture-plan.md`, `lecture-plan-detailed.md`, `interactives-plan.md` | planning docs |
| `teaching-guides/` | per-lecture instructor guides |
| `scripts/` | Typst visual-audit + review tooling, Gemini deck reviewers |
| `_marp-archive/` | the retired Marp system (reference only) |

## Build

Requires Typst ≥ 0.14, IBM Plex Sans/Mono fonts, and the [chalkdust](https://github.com/nipunbatra/chalkdust) packages symlinked into Typst's `@local` package dir (`~/git/chalkdust`).

```bash
make lec5              # one deck (handout) → slides-pdf/L5.pdf
make lec5-pres         # presentation build (each #pause = a page)
make figs-lec5         # regenerate lecture5's matplotlib figures
make slides            # every deck → slides-pdf/  (= ./build-slides-pdf.sh)
make audit             # visual-quality gate over the committed PDFs
quarto render          # the site → _site/ (gitignored)
quarto preview         # local preview
```

Deployment: GitHub Actions (`.github/workflows/quarto-publish.yml`) renders the Quarto site and deploys `_site/` to GitHub Pages on push to master. Slide PDFs are compiled locally and committed — CI runs neither Typst nor Python.

Sibling course repos: [dl-teaching](https://github.com/nipunbatra/dl-teaching) (same Typst system), [ml-teaching](https://github.com/nipunbatra/ml-teaching), [pml-teaching](https://github.com/nipunbatra/pml-teaching), [psdv-teaching](https://github.com/nipunbatra/psdv-teaching).
