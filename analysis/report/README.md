# Proposal (Data Exploration and Methodology and Development Results)

Edit `proposal.tex`; the compiled output is `proposal.pdf`. Data Exploration
occupies pages 1–2 and Methodology and Development Results pages 3–4. The respective limits are
two and three pages. Add group number and author names before submission,
then recheck pagination. Methodology reports completed training, tuning and validation comparison; final test evaluation remains deferred.

From this directory, compile twice to resolve references:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error proposal.tex
pdflatex -interaction=nonstopmode -halt-on-error proposal.tex
pdfinfo proposal.pdf
```

The final command should report `Pages: 4` for this draft. TeX Live supplies the
LaTeX packages; `pdfinfo` is a Poppler utility. The folder is self-contained for
upload to a LaTeX editor: include `proposal.tex` and `figures/`.

Methodology starts on a new page. Current preprocessing and planned changes are explicitly separated.
Statistics come from `../out/proposal/` and `../data_exploration.ipynb`. The payment
figure is copied into `figures/`; update that copy if the analysis changes.
