#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supplemental support tables for the Senegal sorghum hyperspectral manuscript.

This script is the GitHub-ready, non-figure companion for supplemental support analyses.
It recursively discovers project assets, runs supplemental support analyses,
and writes tidy CSV + Excel outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from support_core import (
    AssetDiscovery,
    align_phenotype_and_genotype,
    build_local_region_tables,
    call_qtls,
    compute_correlation_matrices,
    compute_genotype_pca,
    compute_ld_decay,
    discover_assets,
    estimate_ld_window,
    find_trait_column,
    gwas_diagnostic_summary,
    load_assoc_tables_from_discovery,
    load_gene_table,
    load_genotype_and_snpinfo,
    load_interseed_overview,
    load_phenotype_table,
    ld_prune_greedy,
    analysis_coverage_table,
    summarize_interseed_heterogeneity,
    write_csv_bundle,
    write_json,
    write_workbook,
    annotate_qtls_with_genes,
    select_focus_locus,
    ensure_dir,
)

DEFAULT_CORR_TRAITS = [
    "seed_weight_mean",
    "area_mean",
    "circularity_mean",
    "R_total",
    "R_centroid",
    "R_entropy",
    "R_blue_mean",
    "R_green_mean",
    "R_red_mean",
    "R_nir_mean",
    "gray_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate supplemental support tables for the Senegal sorghum spectral manuscript.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Root of the Senegal project folder.")
    parser.add_argument("--output-dir", type=Path, default=Path("revision_outputs") / "tables", help="Directory for CSV/XLSX outputs.")

    # Optional explicit overrides for messy directory layouts
    parser.add_argument("--phenotype-csv", type=Path, default=None)
    parser.add_argument("--supplement-xlsx", type=Path, default=None)
    parser.add_argument("--genotype-csv", type=Path, default=None)
    parser.add_argument("--snp-info-csv", type=Path, default=None)
    parser.add_argument("--hapmap-path", type=Path, default=None)
    parser.add_argument("--gff-path", type=Path, default=None)
    parser.add_argument("--synonym-path", type=Path, default=None)
    parser.add_argument("--defline-path", type=Path, default=None)
    parser.add_argument("--annotation-info-path", type=Path, default=None)
    parser.add_argument("--assoc-dir", type=Path, default=None)

    # LD / locus settings
    parser.add_argument("--ld-max-distance-bp", type=int, default=2_000_000)
    parser.add_argument("--ld-distance-bin-bp", type=int, default=25_000)
    parser.add_argument("--ld-max-snps-per-chr", type=int, default=600)
    parser.add_argument("--ld-threshold-r2", type=float, default=0.20)
    parser.add_argument("--qtl-window-bp", type=int, default=250_000, help="Window used for reviewer-facing QTL grouping.")
    parser.add_argument("--prune-window-bp", type=int, default=250_000)
    parser.add_argument("--prune-r2-threshold", type=float, default=0.80)
    parser.add_argument("--prune-max-snps-per-chr", type=int, default=1500)

    # Focus locus settings
    parser.add_argument("--focus-chrom", default="6")
    parser.add_argument("--focus-trait", default=None, help="Optional explicit focus trait for the local chr6 analysis.")
    parser.add_argument("--focus-lead-snp", default=None)
    parser.add_argument("--focus-pos", type=int, default=None)
    parser.add_argument("--local-window-bp", type=int, default=250_000)
    parser.add_argument("--local-ld-max-snps", type=int, default=80)
    parser.add_argument("--haplotype-max-snps", type=int, default=4)
    parser.add_argument("--haplotype-r2-threshold", type=float, default=0.60)

    parser.add_argument("--corr-traits", nargs="*", default=DEFAULT_CORR_TRAITS)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-workbook", action="store_true", help="Write CSV/JSON outputs only (useful for fast validation).")
    parser.add_argument("--full-workbook", action="store_true", help="Include long tables in the Excel workbook; by default the workbook is summary-focused and long tables remain in CSV.")
    return parser.parse_args()



def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)



def build_qtl_tables(assoc_tables: Dict[str, pd.DataFrame], gene_df: pd.DataFrame, qtl_window_bp: int) -> Dict[str, pd.DataFrame]:
    all_qtls: List[pd.DataFrame] = []
    for trait, df in assoc_tables.items():
        qtl = call_qtls(df, window_bp=qtl_window_bp)
        if not qtl.empty:
            all_qtls.append(qtl)
    if not all_qtls:
        return {"qtl_peak_summary": pd.DataFrame(), "qtl_gene_long": pd.DataFrame()}
    qtl_all = pd.concat(all_qtls, ignore_index=True).sort_values(["trait", "chrom", "lead_pos"]).reset_index(drop=True)
    qtl_summary, qtl_gene_long = annotate_qtls_with_genes(qtl_all, gene_df)
    return {
        "qtl_peak_summary": qtl_summary,
        "qtl_gene_long": qtl_gene_long,
    }



def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = ensure_dir(args.output_dir.resolve())
    csv_dir = ensure_dir(output_dir / "csv")

    log(f"[1/8] Discovering assets under: {project_root}", args.verbose)
    assets: AssetDiscovery = discover_assets(
        project_root=project_root,
        phenotype_csv=args.phenotype_csv,
        supplement_xlsx=args.supplement_xlsx,
        genotype_csv=args.genotype_csv,
        snp_info_csv=args.snp_info_csv,
        hapmap_path=args.hapmap_path,
        gff_path=args.gff_path,
        synonym_path=args.synonym_path,
        defline_path=args.defline_path,
        annotation_info_path=args.annotation_info_path,
        assoc_dir=args.assoc_dir,
    )
    manifest_df = assets.to_manifest_records()

    log("[2/8] Loading phenotype, genotype, and gene annotation", args.verbose)
    phenotype_df, phenotype_source = load_phenotype_table(assets)
    geno_df, snp_info, genotype_source = load_genotype_and_snpinfo(assets)
    gene_df, gene_source = load_gene_table(assets)
    phenotype_df, geno_df = align_phenotype_and_genotype(phenotype_df, geno_df)

    shared_snps = [s for s in snp_info.index if s in geno_df.columns]
    snp_info = snp_info.loc[shared_snps].copy()
    geno_df = geno_df[shared_snps].copy()

    sources_df = pd.DataFrame([
        {"item": "phenotype_source", "value": phenotype_source},
        {"item": "genotype_source", "value": genotype_source},
        {"item": "gene_source", "value": gene_source},
        {"item": "n_accessions_overlap", "value": phenotype_df.shape[0]},
        {"item": "n_snps_overlap", "value": geno_df.shape[1]},
        {"item": "qtl_window_bp", "value": args.qtl_window_bp},
        {"item": "local_window_bp", "value": args.local_window_bp},
    ])

    log("[3/8] Running LD decay and pruning summaries", args.verbose)
    ld_pairs, ld_bins = compute_ld_decay(
        geno_df=geno_df,
        snp_info=snp_info,
        max_distance_bp=args.ld_max_distance_bp,
        distance_bin_bp=args.ld_distance_bin_bp,
        max_snps_per_chr=args.ld_max_snps_per_chr,
    )
    ld_summary = estimate_ld_window(ld_bins, threshold_r2=args.ld_threshold_r2)
    prune_summary, pruned_snps = ld_prune_greedy(
        geno_df=geno_df,
        snp_info=snp_info,
        window_bp=args.prune_window_bp,
        r2_threshold=args.prune_r2_threshold,
        max_snps_per_chr=args.prune_max_snps_per_chr,
    )
    prune_overall = pd.DataFrame([{
        "n_total_snps": int(geno_df.shape[1]),
        "n_pruned_snps_kept": int(len(pruned_snps)),
        "fraction_kept": float(len(pruned_snps) / max(geno_df.shape[1], 1)),
        "window_bp": int(args.prune_window_bp),
        "r2_threshold": float(args.prune_r2_threshold),
    }])

    log("[4/8] Computing genotype PCA scree and correlation heatmap tables", args.verbose)
    _, geno_scree = compute_genotype_pca(geno_df, n_components=20)
    corr_r, corr_p, corr_long = compute_correlation_matrices(phenotype_df, args.corr_traits)

    log("[5/8] Loading GWAS association tables and QTL summaries", args.verbose)
    assoc_tables = load_assoc_tables_from_discovery(assets)
    gwas_diag = gwas_diagnostic_summary(assoc_tables)
    qtl_tables = build_qtl_tables(assoc_tables, gene_df, qtl_window_bp=args.qtl_window_bp)

    log("[6/8] Building chr6 local-region support tables", args.verbose)
    if args.focus_trait and args.focus_trait in assoc_tables:
        focus_trait = args.focus_trait
        lead_row = assoc_tables[focus_trait].loc[assoc_tables[focus_trait]["p"].idxmin()]
    else:
        focus_trait, lead_row = select_focus_locus(assoc_tables, focus_chrom=args.focus_chrom)
    if focus_trait is not None and lead_row is not None:
        focus_assoc = assoc_tables[focus_trait]
        local_tables = build_local_region_tables(
            assoc_df=focus_assoc,
            geno_df=geno_df,
            snp_info=snp_info,
            gene_df=gene_df,
            phenotype_df=phenotype_df,
            focus_trait=focus_trait,
            focus_lead_snp=args.focus_lead_snp,
            focus_chrom=args.focus_chrom,
            focus_pos=args.focus_pos,
            window_bp=args.local_window_bp,
            ld_max_snps=args.local_ld_max_snps,
            haplotype_max_snps=args.haplotype_max_snps,
            haplotype_r2_threshold=args.haplotype_r2_threshold,
        )
    else:
        local_tables = {
            "local_assoc": pd.DataFrame(),
            "local_ld": pd.DataFrame(),
            "local_genes": pd.DataFrame(),
            "haplotypes": pd.DataFrame(),
            "haplotype_trait_summary": pd.DataFrame(),
            "local_summary": pd.DataFrame(),
        }

    # Add phenotype alias info for the focus trait (useful for boxplots/manuscript table)
    focus_trait_col = find_trait_column(phenotype_df, focus_trait) if focus_trait else None
    focus_info = pd.DataFrame([{
        "focus_trait": focus_trait or "",
        "focus_trait_phenotype_column": focus_trait_col or "",
        "requested_focus_snp": args.focus_lead_snp or "",
        "requested_focus_pos": args.focus_pos if args.focus_pos is not None else "",
    }])

    log("[7/8] Supplementary inter-seed and heterogeneity summaries", args.verbose)
    heterogeneity_summary = summarize_interseed_heterogeneity(phenotype_df)
    interseed_overview = load_interseed_overview(assets)

    # Optional compact top-peak table for point-by-point response drafting
    top_peak_table = pd.DataFrame()
    if not qtl_tables["qtl_peak_summary"].empty:
        top_peak_table = qtl_tables["qtl_peak_summary"].copy()
        top_peak_table = top_peak_table.sort_values(["lead_p", "trait"]).groupby("trait", as_index=False).head(3).reset_index(drop=True)

    sheet_dict = {
        "analysis_map": analysis_coverage_table(),
        "sources": sources_df,
        "manifest": manifest_df,
        "ld_decay_bins": ld_bins,
        "ld_decay_summary": ld_summary,
        "ld_pruning_by_chr": prune_summary,
        "ld_pruning_overall": prune_overall,
        "geno_pca_scree": geno_scree,
        "gwas_diagnostics": gwas_diag,
        "qtl_peak_summary": qtl_tables["qtl_peak_summary"],
        "qtl_gene_long": qtl_tables["qtl_gene_long"],
        "top_peak_table": top_peak_table,
        "chr6_local_summary": local_tables["local_summary"],
        "chr6_focus_info": focus_info,
        "chr6_local_assoc": local_tables["local_assoc"],
        "chr6_local_ld": local_tables["local_ld"],
        "chr6_local_genes": local_tables["local_genes"],
        "chr6_haplotypes": local_tables["haplotypes"],
        "chr6_haplotype_trait_summary": local_tables["haplotype_trait_summary"],
        "correlation_r": corr_r,
        "correlation_p": corr_p,
        "correlation_long": corr_long,
        "heterogeneity_summary": heterogeneity_summary,
        "interseed_overview": interseed_overview,
    }

    log("[8/8] Writing CSV/XLSX outputs", args.verbose)
    long_sheet_names = {"qtl_gene_long"}
    workbook_sheet_dict = sheet_dict if args.full_workbook else {k: v for k, v in sheet_dict.items() if k not in long_sheet_names}
    write_csv_bundle(csv_dir, sheet_dict)
    workbook_path = output_dir / "senegal_support_tables.xlsx"
    if not args.skip_workbook:
        write_workbook(workbook_path, workbook_sheet_dict)
    write_json(output_dir / "run_manifest.json", {
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "phenotype_source": phenotype_source,
        "genotype_source": genotype_source,
        "gene_source": gene_source,
        "focus_trait": focus_trait,
        "focus_trait_column": focus_trait_col,
        "assoc_traits": sorted(assoc_tables.keys()),
        "qtl_window_bp": args.qtl_window_bp,
        "local_window_bp": args.local_window_bp,
        "ld_threshold_r2": args.ld_threshold_r2,
        "skip_workbook": bool(args.skip_workbook),
        "full_workbook": bool(args.full_workbook),
        "long_tables_csv_only": sorted(long_sheet_names) if not args.full_workbook else [],
    })

    print("\n[OK] Supplemental support table bundle created:")
    if args.skip_workbook:
        print("  Excel workbook : skipped (--skip-workbook)")
    else:
        print(f"  Excel workbook : {workbook_path}")
    print(f"  CSV folder     : {csv_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
