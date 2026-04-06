# ClusterSuit Thesis - LaTeX Template

This directory contains the LaTeX source files for the ClusterSuit thesis, formatted according to City University of Hong Kong School of Data Science guidelines.

## File Structure

```
.
├── thesis.tex          # Main LaTeX document
├── references.bib      # Bibliography file
├── figures/            # Figure directory (symlink to GhostSuit_2/figures)
└── README.md           # This file
```

## Requirements

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages (included in standard distributions):
  - geometry
  - setspace
  - times
  - graphicx
  - amsmath, amssymb, amsthm
  - hyperref
  - booktabs
  - longtable
  - algorithm, algorithmicx, algpseudocode
  - listings
  - xcolor
  - enumitem

## Compilation

To compile the thesis, run the following commands in order:

```bash
# First compilation
pdflatex thesis.tex

# Compile bibliography
bibtex thesis

# Second compilation (for references)
pdflatex thesis.tex

# Third compilation (for TOC, LOA, LOT)
pdflatex thesis.tex
```

Or use `latexmk`:

```bash
latexmk -pdf thesis.tex
```

## Figures

The `figures/` directory should contain:
- Figure_1_Architecture.png
- Figure_2_Layer_Time.png
- Figure_3_Cluster_Smoothing.png
- Figure_4a_AUROC.png
- Figure_4b_Throughput.png
- Figure_5_Alpha_Sensitivity.png
- Figure_6_GPT2_vs_ResNet.png
- Figure_7_TopK_Recall.png
- Figure_8_Tradeoff.png

These are automatically loaded from `../figures/` relative to the thesis.tex location.

## Document Structure

The thesis follows the standard CityU SDSC thesis format:

1. Title Page
2. Declaration Page
3. Abstract
4. Acknowledgements
5. Table of Contents
6. List of Figures
7. List of Tables
8. Chapter 1: Introduction
9. Chapter 2: Background and Related Work
10. Chapter 3: Methodology
11. Chapter 4: Experiments
12. Chapter 5: Discussion and Conclusion
13. References
14. Appendix

## Troubleshooting

### Missing packages
Install missing packages using your LaTeX distribution's package manager.

### Figure not found
Ensure the `figures/` directory exists and contains all required PNG files. The path is relative to the thesis.tex file.

### Bibliography errors
Make sure to run `bibtex thesis` between the first and second `pdflatex` runs.
