# -------------------------------------------------------
# Makefile for Quantum–Reflexive Control Framework Project
# -------------------------------------------------------

TEX = main.tex
OUTDIR = build
LOGDIR = $(OUTDIR)/logs
PDF = $(OUTDIR)/main.pdf

LATEXMK = latexmk
LATEXMK_OPTS = -pdf -interaction=nonstopmode -halt-on-error \
               -outdir=$(OUTDIR) -shell-escape

default: pdf

pdf:
	mkdir -p $(OUTDIR)
	mkdir -p $(LOGDIR)
	$(LATEXMK) $(LATEXMK_OPTS) $(TEX) | tee $(LOGDIR)/build.log

clean:
	$(LATEXMK) -C -outdir=$(OUTDIR)
	rm -rf $(LOGDIR)

full-clean:
	rm -rf $(OUTDIR)

watch:
	$(LATEXMK) $(LATEXMK_OPTS) -pvc $(TEX)

status:
	@echo "PDF: $(PDF)"
	@echo "Log directory: $(LOGDIR)"

.PHONY: pdf clean full-clean watch status
