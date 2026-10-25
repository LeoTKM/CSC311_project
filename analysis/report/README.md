# Proposal (Data Exploration and Methodology and Development Results)

Edit `proposal.tex`; the compiled output is `proposal.pdf`. Data Exploration
occupies pages 1–2, Methodology and Development Results pages 3–4, and References
page 5. The report template specifies at most two pages for Data Exploration
and three for Methodology; the Proposal instructions do not specify a separate
overall page limit. Group number and member names are already filled in; verify
them before submission. Methodology reports completed training, tuning and
validation comparison; final test evaluation remains deferred.

From this directory, compile twice to resolve references:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error proposal.tex
pdflatex -interaction=nonstopmode -halt-on-error proposal.tex
pdfinfo proposal.pdf
```

The final command should report `Pages: 5` for this draft. TeX Live supplies the
LaTeX packages; `pdfinfo` is a Poppler utility. The folder is self-contained for
upload to a LaTeX editor: include `proposal.tex` and `figures/`.

Methodology starts on a new page. Current preprocessing and planned changes are explicitly separated.
Statistics come from `../out/proposal/` and `../data_exploration.ipynb`. The payment
figure is copied into `figures/`; update that copy if the analysis changes.

## Prepare the submission

From the repository root, run:

```bash
python analysis/package_proposal.py
```

This creates `submission/proposal.pdf` (an exact copy of the compiled report)
and `submission/code.zip`. The ZIP contains the Python scripts, notebooks,
dependency list and code READMEs from this branch. It excludes `/data`, generated
`analysis/out` artifacts, saved models, environments and caches. Notebook outputs
are retained as evidence of the work. The archive is checked for corruption.
Generated submission files are ignored by Git; upload them to MarkUs separately.
To rebuild, choose a fresh `--output-dir`; existing deliverables are not overwritten.
