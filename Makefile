# MFAI course build — Typst decks + Quarto site
#
#   make lec5          compile lecture5's deck (handout) -> slides-pdf/L5.pdf
#   make lec5-pres     presentation build (each #pause = a page) -> /tmp/L5-presentation.pdf
#   make lec5-watch    typst watch (handout) -> /tmp/L5.pdf
#   make figs-lec5     regenerate lecture5's matplotlib figures (if any)
#   make slides        compile every deck -> slides-pdf/
#   make audit         visual-quality gate over every committed PDF
#   make render        quarto render the site -> _site/
#   make preview       quarto preview
#
# Decks live in lectureN/LN-<slug>.typ; figure scripts in lectureN/diagrams/lN_figs.py.
# Figure paths inside decks are root-absolute, so --root . is mandatory.

SHELL := /bin/bash
PY_FIGS := uv run --no-project --with matplotlib,numpy,scikit-learn python3

.PHONY: slides audit render preview clean list

slides:
	./build-slides-pdf.sh

audit:
	python3 scripts/audit_typst_slides.py slides-pdf/L*.pdf

render:
	quarto render

preview:
	quarto preview

list:
	@ls lecture*/L*.typ 2>/dev/null | sed 's/^/  /'

clean:
	rm -rf _site

# ── per-lecture targets: make lec5 / lec5-pres / lec5-watch / figs-lec5 ──
lec%: lecture%/
	@deck=$$(ls lecture$*/L*.typ | head -1); \
	 out="slides-pdf/L$*.pdf"; \
	 echo "  $$deck -> $$out (handout)"; \
	 mkdir -p slides-pdf; \
	 typst compile --root . --input handout=true "$$deck" "$$out"

lec%-pres: lecture%/
	@deck=$$(ls lecture$*/L*.typ | head -1); \
	 out="/tmp/L$*-presentation.pdf"; \
	 echo "  $$deck -> $$out (presentation)"; \
	 typst compile --root . "$$deck" "$$out"

lec%-watch: lecture%/
	@deck=$$(ls lecture$*/L*.typ | head -1); \
	 echo "  watching $$deck -> /tmp/L$*.pdf"; \
	 typst watch --root . --input handout=true "$$deck" /tmp/L$*.pdf

figs-lec%: lecture%/
	@script=$$(ls lecture$*/diagrams/l*_figs.py 2>/dev/null | head -1); \
	 if [ -z "$$script" ]; then echo "no figure script in lecture$*/diagrams/"; exit 1; fi; \
	 echo "  running $$script"; \
	 $(PY_FIGS) "$$script"

# directory prerequisites are just existence checks
lecture%/: ;
