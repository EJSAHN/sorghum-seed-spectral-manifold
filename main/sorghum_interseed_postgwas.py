#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sorghum seed inter-seed distribution GWAS post-processing.

This script performs downstream post-GWAS processing for inter-seed optical
heterogeneity traits using existing GEMMA association outputs and gene summary
files produced by the main manuscript analysis.

The script is designed to run from a project root directory without
hard-coded machine-specific paths.
"""

from __future__ import annotations

import argparse
import os
import glob
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run inter-seed GWAS post-processing for the Senegal sorghum spectral manuscript.'
    )
    parser.add_argument('--project-root', default='.', help='Project root directory containing manuscript outputs.')
    parser.add_argument('--output-subdir', default='_paper1_output_spectral_v5', help='Main output subdirectory under the project root.')
    parser.add_argument('--annotation-dir', default='.', help='Directory containing Phytozome-style annotation/defline files. Relative to project root unless absolute.')
    return parser.parse_args()


ARGS = parse_args()
BASE_DIR = os.path.abspath(ARGS.project_root)
OUT_DIR = os.path.join(BASE_DIR, ARGS.output_subdir)
GEMMA_OUT_DIR = os.path.join(OUT_DIR, 'output')
GENE_SUMMARY_FILE = os.path.join(OUT_DIR, 'annotation', 'annotated_gwas_gene_summary.csv')
REF_DIR = ARGS.annotation_dir if os.path.isabs(ARGS.annotation_dir) else os.path.join(BASE_DIR, ARGS.annotation_dir)
QTL_OUT_FILE = os.path.join(OUT_DIR, 'QTL_interseed_traits_GEMMA.csv')
CAND_OUT_FILE = os.path.join(OUT_DIR, 'Candidate_Genes_InterSeed_Annotated.csv')
CAND_DEFLINE_OUT_FILE = os.path.join(OUT_DIR, 'Candidate_Genes_InterSeed_Annotated_withDefline.csv')

print('BASE_DIR      :', BASE_DIR)
print('OUT_DIR       :', OUT_DIR)
print('GEMMA_OUT_DIR :', GEMMA_OUT_DIR)
print('GENE_SUMMARY  :', GENE_SUMMARY_FILE)
print('REF_DIR       :', REF_DIR)

# =====================================================================
# [1] Load GEMMA association files for inter-seed traits
# =====================================================================

def load_gemma_assoc(filename, gemma_dir=GEMMA_OUT_DIR):
    """
    Load a GEMMA .assoc.txt file and return a DataFrame with columns:
      [CHR, BP, P, SNP]
    """
    # resolve path
    if not os.path.isabs(filename):
        path = os.path.join(gemma_dir, filename)
    else:
        path = filename

    if not os.path.exists(path):
        # try with common extensions
        for ext in [".assoc.txt", ".assoc"]:
            if os.path.exists(path + ext):
                path = path + ext
                break

    if not os.path.exists(path):
        raise FileNotFoundError(f"GEMMA assoc file not found: {path}")

    df = pd.read_csv(path, delim_whitespace=True)
    col_map = {c.lower(): c for c in df.columns}

    chr_col = col_map.get("chr")
    bp_col = (col_map.get("ps") or col_map.get("bp") or
              col_map.get("pos") or col_map.get("position"))
    p_col = (col_map.get("p_wald") or col_map.get("p_lrt") or
             col_map.get("p_score") or col_map.get("p_value"))
    rs_col = (col_map.get("rs") or col_map.get("snp") or
              col_map.get("marker"))

    if chr_col is None or bp_col is None or p_col is None:
        raise ValueError(f"Could not find chr/position/p columns in: {df.columns}")

    out = pd.DataFrame()
    out["CHR"] = df[chr_col].astype(int)
    out["BP"] = df[bp_col].astype(int)
    out["P"] = df[p_col].astype(float)

    if rs_col is not None:
        out["SNP"] = df[rs_col].astype(str)
    else:
        # if no rsID, synthesize one
        out["SNP"] = out.apply(lambda r: f"chr{r['CHR']}_{r['BP']}", axis=1)

    out = out.sort_values(["CHR", "BP"]).reset_index(drop=True)
    return out


# Map inter-seed traits to their GEMMA output files
INTERSEED_TRAIT_FILES = {
    "InterSeed_R750_Std":  "GWAS_InterSeed_R750_Std.assoc.txt",
    "InterSeed_F450_Skew": "GWAS_InterSeed_F450_Skew.assoc.txt",
    # add or comment out traits as needed, e.g.:
    # "InterSeed_R750_p10": "GWAS_InterSeed_R750_p10.assoc.txt",
    # "InterSeed_R650_Std": "GWAS_InterSeed_R650_Std.assoc.txt",
}


# =====================================================================
# [2] QTL calling per trait (±250 kb window)
# =====================================================================

def call_qtls_for_trait(df_assoc, trait_name,
                        window_kb=250,
                        alpha=0.05):
    """
    For a single trait:
    - Apply Bonferroni threshold alpha / M
    - Group Bonferroni-significant SNPs on each chromosome into QTLs
      using a ±window_kb window around the lead SNP.
    """
    m = df_assoc.shape[0]
    bonf_p = alpha / m

    sig = df_assoc[df_assoc["P"] <= bonf_p].copy()
    if sig.empty:
        print(f"[{trait_name}] No Bonferroni-significant SNPs (M={m}, alpha={alpha})")
        return []

    window_bp = window_kb * 1000
    qtl_rows = []

    for chr_ in sorted(sig["CHR"].unique()):
        sub = sig[sig["CHR"] == chr_].sort_values("BP").reset_index(drop=True)
        bps = sub["BP"].to_numpy()
        ps = sub["P"].to_numpy()
        snps = sub["SNP"].to_numpy()

        i = 0
        while i < len(sub):
            cluster_start = bps[i]
            j = i + 1
            while j < len(sub) and (bps[j] - cluster_start) <= window_bp:
                j += 1

            # cluster i..j-1 is one QTL
            cluster_slice = slice(i, j)
            cluster_ps = ps[cluster_slice]
            lead_rel = np.argmin(cluster_ps)
            lead_idx = i + lead_rel

            lead_bp = int(bps[lead_idx])
            lead_p = float(ps[lead_idx])
            lead_snp = snps[lead_idx]

            qtl_start = int(max(0, lead_bp - window_bp))
            qtl_end = int(lead_bp + window_bp)

            qtl_rows.append({
                "trait": trait_name,
                "chr": int(chr_),
                "qtl_start": qtl_start,
                "qtl_end": qtl_end,
                "lead_snp": lead_snp,
                "lead_bp": lead_bp,
                "lead_p": lead_p,
                "n_sig_snps": int(j - i),
                "bonf_p": bonf_p,
            })

            i = j

    return qtl_rows


def run_qtl_call_all_traits(trait_file_map=INTERSEED_TRAIT_FILES,
                            out_csv=QTL_OUT_FILE):
    all_rows = []

    for trait, fname in trait_file_map.items():
        print(f"\n=== QTL calling for {trait} ({fname}) ===")
        assoc = load_gemma_assoc(fname)
        print(f"  - Number of SNPs: {assoc.shape[0]}")
        rows = call_qtls_for_trait(assoc, trait_name=trait)
        all_rows.extend(rows)

    if not all_rows:
        print("\n[Warning] No significant QTL detected for any trait.")
        return None

    qtl_df = pd.DataFrame(all_rows)
    qtl_df = qtl_df.sort_values(["trait", "chr", "lead_bp"]).reset_index(drop=True)
    qtl_df.to_csv(out_csv, index=False)
    print(f"\n[OK] Inter-seed QTL summary saved: {out_csv}")
    print(qtl_df.head())
    return qtl_df


# =====================================================================
# [3] Load gene DB from main annotated gene summary
# =====================================================================

def load_gene_db_from_summary(gene_summary_file=GENE_SUMMARY_FILE):
    if not os.path.exists(gene_summary_file):
        raise FileNotFoundError(f"Gene summary file not found: {gene_summary_file}")

    df = pd.read_csv(gene_summary_file)
    print(f"[Info] Gene summary loaded: {df.shape[0]} rows")

    col_map = {c.lower(): c for c in df.columns}

    chr_col = (col_map.get("gene_chrom") or col_map.get("chrom") or
               col_map.get("chr"))
    start_col = col_map.get("gene_start") or col_map.get("start")
    end_col = col_map.get("gene_end") or col_map.get("end")
    id_col = (col_map.get("locus") or col_map.get("gene") or
              col_map.get("gene_id"))
    desc_col = (col_map.get("sorghum_defline") or
                col_map.get("defline") or
                col_map.get("description"))

    if not (chr_col and start_col and end_col and id_col):
        raise ValueError("Could not find chr/start/end/id columns in gene summary.")

    gene = df.copy()

    # chr: e.g. Chr01 -> 1
    gene["chr"] = (gene[chr_col].astype(str)
                   .str.extract(r"(\d+)")[0]
                   .astype("Int64"))
    gene["start"] = pd.to_numeric(gene[start_col], errors="coerce").astype("Int64")
    gene["end"] = pd.to_numeric(gene[end_col], errors="coerce").astype("Int64")
    gene["gene_id"] = gene[id_col].astype(str)

    if desc_col:
        gene["description"] = gene[desc_col].astype(str)
    else:
        gene["description"] = "No description"

    gene = gene.dropna(subset=["chr", "start", "end"])
    gene = gene.drop_duplicates(subset=["gene_id"])

    gene = gene[["chr", "start", "end", "gene_id", "description"]].reset_index(drop=True)
    print(f"[OK] Gene DB built: {gene.shape[0]} genes")
    return gene


# =====================================================================
# [4] QTL × Gene intersection → Candidate gene table
# =====================================================================

def annotate_interseed_qtls(qtl_df,
                            gene_df,
                            out_csv=CAND_OUT_FILE):
    results = []

    print("\n[Step] Intersecting QTL intervals with gene coordinates...")
    for _, qtl in qtl_df.iterrows():
        chr_ = int(qtl["chr"])
        qtl_start = int(qtl["qtl_start"])
        qtl_end = int(qtl["qtl_end"])

        candidates = gene_df[
            (gene_df["chr"] == chr_) &
            (gene_df["start"] <= qtl_end) &
            (gene_df["end"] >= qtl_start)
        ].copy()

        if candidates.empty:
            continue

        for _, gene in candidates.iterrows():
            dist = abs(int(gene["start"]) - int(qtl["lead_bp"]))
            results.append({
                "Trait": qtl["trait"],
                "Chr": chr_,
                "QTL_Lead_SNP": qtl["lead_snp"],
                "QTL_Lead_BP": int(qtl["lead_bp"]),
                "QTL_P_value": float(qtl["lead_p"]),
                "Gene_ID": gene["gene_id"],
                "Gene_Start": int(gene["start"]),
                "Gene_End": int(gene["end"]),
                "Distance_from_Lead": dist,
                "Description": gene["description"],
            })

    if not results:
        print("[Warning] No candidate genes found within QTL windows.")
        return None

    cand_df = pd.DataFrame(results)
    cand_df = cand_df.sort_values(
        ["Trait", "Chr", "QTL_Lead_BP", "Distance_from_Lead"]
    ).reset_index(drop=True)

    cand_df.to_csv(out_csv, index=False)
    print(f"[OK] Candidate gene table saved: {out_csv}")
    print(cand_df.head())
    return cand_df


# =====================================================================
# [5] Load defline table and fill descriptions
# =====================================================================

def load_defline_table(ref_dir=REF_DIR):
    """
    Try to locate a Phytozome-style annotation_info/defline file
    under ref_dir and return a simple [Gene_ID_Base, Defline] table.
    """
    candidates = glob.glob(os.path.join(ref_dir, "*annotation_info*.txt*")) + \
                 glob.glob(os.path.join(ref_dir, "*defline*.txt*"))

    if not candidates:
        print("[Warning] No annotation_info/defline file found.")
        return None

    path = candidates[0]
    comp = "gzip" if path.endswith(".gz") else None
    print(f"[Info] Using defline file: {path}")

    try:
        df = pd.read_csv(path, sep="\t", compression=comp, low_memory=False)
    except Exception as e:
        print("  [Warning] Failed with tab separator, retrying with whitespace:", e)
        df = pd.read_csv(path, sep=r"\s+", compression=comp, low_memory=False)

    col_map = {str(c).lower(): c for c in df.columns}
    id_col = (col_map.get("locusname") or
              col_map.get("gene") or
              col_map.get("id") or
              df.columns[0])
    desc_col = (col_map.get("best-hit-rice-defline") or
                col_map.get("best-hit-arabi-defline") or
                col_map.get("defline") or
                col_map.get("note") or
                df.columns[-1])

    defline = df[[id_col, desc_col]].copy()
    defline.columns = ["Gene_ID_raw", "Defline"]

    # e.g. Sobic.001G000100.1 -> Sobic.001G000100
    defline["Gene_ID_Base"] = (defline["Gene_ID_raw"].astype(str)
                               .str.split(".").str[:2].str.join("."))
    defline = defline.drop_duplicates(subset=["Gene_ID_Base"])
    print(f"[OK] Defline table: {defline.shape[0]} genes")
    return defline


def add_defline_to_candidates(cand_df,
                              ref_dir=REF_DIR,
                              out_csv=CAND_DEFLINE_OUT_FILE):
    defline = load_defline_table(ref_dir=ref_dir)
    if defline is None:
        print("[Warning] Skipping defline merge; writing original candidate table.")
        cand_df.to_csv(out_csv, index=False)
        return cand_df

    df = cand_df.copy()
    df["Gene_ID_Base"] = df["Gene_ID"].astype(str).str.split(".").str[:2].str.join(".")

    merged = df.merge(defline[["Gene_ID_Base", "Defline"]],
                      on="Gene_ID_Base",
                      how="left")

    # Replace empty/placeholder descriptions with Defline
    desc = merged["Description"].astype(str).str.strip()
    cond_empty = desc.isna() | desc.isin(["", "No description", "nan", "None"])

    n_before = cond_empty.sum()
    merged.loc[cond_empty, "Description"] = merged.loc[cond_empty, "Defline"]
    n_after = merged["Description"].isna().sum()

    merged = merged.drop(columns=["Gene_ID_Base", "Defline"])
    merged.to_csv(out_csv, index=False)

    print(f"\n[OK] Defline merge completed: {out_csv}")
    print(f"   - rows with empty description before: {n_before}")
    print(f"   - rows still empty after merge:      {n_after}")
    print(merged.head())

    return merged


# =====================================================================
# [6] Main entry point
# =====================================================================

if __name__ == "__main__":
    # 1) Call QTLs for all inter-seed traits
    qtl_df = run_qtl_call_all_traits()

    if qtl_df is not None:
        # 2) Load gene DB from main annotated summary
        gene_db = load_gene_db_from_summary()

        # 3) QTL × gene intersection
        cand_df = annotate_interseed_qtls(qtl_df, gene_db)

        if cand_df is not None:
            # 4) Enrich descriptions with defline information
            cand_def = add_defline_to_candidates(cand_df)
