# Supplementary Software 1: Sorghum spectral pipeline

This package contains the cleaned analysis code used to support the manuscript
**"Decoupling the chemical and physical origins of the seed spectral manifold in sorghum"**.

The package is designed to reproduce manuscript-level downstream analyses from
curated phenotype/genotype inputs and GWAS summary files.

## Package layout

- `main/sorghum_spectral_main.py`
  - Core manuscript analysis pipeline
  - Builds spectral summary matrices from processed inputs
  - Merges morphology and spectral traits
  - Constructs genotype PCs
  - Prepares GEMMA inputs and summarizes GWAS/QTL results
  - Runs path analysis and in silico manifold perturbation
  - Annotates GWAS summaries with sorghum gene models

- `main/sorghum_interseed_postgwas.py`
  - Post-GWAS processing for inter-seed optical heterogeneity traits
  - Summarizes QTLs from GEMMA association outputs
  - Maps intervals to candidate genes
  - Enriches descriptions with annotation/defline resources

- `support/support_tables.py`
  - Supplemental analysis tables (tables only; no figure generation)
  - Computes empirical LD decay, genotype PCA scree, chromosome 6 local summaries,
    locus-level peak tables, and related support outputs

- `support/support_core.py`
  - Shared utilities for the supplemental support analyses

## What is intentionally *not* included

- Figure-generation/refinement scripts used only for manuscript layout
- Exploratory or troubleshooting notebooks
- Raw hyperspectral image cubes
- Platform-specific executables (for example, PLINK or GEMMA binaries)

## Data sources expected by this package

Please see `docs/input_data_sources.md` for the full list. In brief:

- Seed morphology phenotype data: previously published supplementary data
- SNP genotype data: public Figshare repository
- Curated spectral and GWAS summary tables: Supplementary Data S1 of the manuscript
- Raw hyperspectral image data: available from the corresponding author upon reasonable request

## External tools

This package does **not** bundle GEMMA or PLINK.

- The core manuscript script prepares GEMMA input files and summarizes outputs
  after external GEMMA execution.
- The supplemental support script works directly from curated project files and
  existing GWAS outputs; PLINK is **not required** for normal reruns.

## Quick start

See `run_examples.txt` for Windows-ready example commands.

## Notes

This package is intentionally focused on reproducible downstream analyses from
processed inputs. It is suitable for manuscript-associated sharing and can be
adapted directly into a public GitHub repository if desired.
