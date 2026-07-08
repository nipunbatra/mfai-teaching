# Mathematical Foundations for AI — IIT Gandhinagar

Course site: https://nipunbatra.github.io/mfai-teaching · Prof. Nipun Batra

26 lectures + 13 hybrid tutorials (80 min each). Floating point → linear algebra → multivariate calculus & autodiff → probability & estimation → optimization → information theory → Markov chains. The finale trains a character-level n-gram language model that uses every module.

## Repo layout

| Path | What |
|------|------|
| `slides/lecNN-*-lecture.md` | Marp decks (theme: `slides/mfai-theme.css`) |
| `diagrams/lecNN_figures.py` | matplotlib scripts → `figures/lecNN/` (PNG) + `figures/lecNN/svg/` |
| `notebooks/` | lecture companions (`lecNN-*`) and tutorial notebooks (`tutNN-*`) |
| `tutorials/` | worksheet pages (Quarto) |
| `*.qmd` | Quarto site pages |
| `course-design.md`, `lecture-plan.md`, `lecture-plan-detailed.md`, `interactives-plan.md` | planning docs |
| `teaching-guides/` | per-lecture instructor guides |
| `scripts/` | Gemini-based deck review tooling |

## Build

```bash
npm ci                 # marp-cli + puppeteer
make lec01             # one deck → html/ + pdf/
make first N=6         # HTML for L1–L6
make all               # everything (runs diagrams first)
make diagrams          # regenerate figures
quarto render          # the site (html/ and pdf/ are outputs, gitignored)
quarto preview         # local preview
```

Deployment: GitHub Actions (`.github/workflows/quarto-publish.yml`) builds Marp HTML+PDF, renders Quarto, and deploys the repo root to GitHub Pages on push to master. Only sources are committed; all rendered output is gitignored.

Sibling course repos following the same system: [dl-teaching](https://github.com/nipunbatra/dl-teaching), [ml-teaching](https://github.com/nipunbatra/ml-teaching), [pml-teaching](https://github.com/nipunbatra/pml-teaching), [psdv-teaching](https://github.com/nipunbatra/psdv-teaching).
