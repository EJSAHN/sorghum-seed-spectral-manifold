#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sorghum seed spectral manifold and GWAS analysis.

This script implements the manuscript-level downstream analysis pipeline used
for the Senegal sorghum spectral manuscript. It is designed to run from a
project root directory without hard-coded machine-specific paths.

The script can:
  - build spectral summary matrices from processed hyperspectral summaries
  - merge morphology and spectral traits into a seed phenome table
  - construct genotype matrices and genotype PCs
  - quantify genomic structure-associated spectral variance
  - prepare GEMMA input files and summarize association outputs
  - run path analysis and in silico manifold perturbation
  - annotate GWAS summaries using sorghum gene models

Notes
-----
- Raw hyperspectral image cubes are not bundled with this package.
- The script reproduces manuscript-level curated analyses from processed inputs.
- GEMMA is not bundled here; this script prepares GEMMA inputs and summarizes
  outputs after external GEMMA execution.
"""

from __future__ import annotations

import argparse
import os
import gzip
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the core Senegal sorghum spectral manifold analysis pipeline.'
    )
    parser.add_argument('--project-root', default='.', help='Project root directory containing the processed inputs.')
    parser.add_argument('--output-subdir', default='_paper1_output_spectral_v5', help='Output subdirectory under the project root.')
    parser.add_argument('--reflectance-dir', default='R_spectra', help='Directory (under project root) containing processed reflectance summaries.')
    parser.add_argument('--fluorescence-dir', default='F_spectra', help='Directory (under project root) containing processed fluorescence summaries.')
    parser.add_argument('--hapmap-file', default='combined_Anthracnosemarch.hmp.txt', help='HapMap genotype file relative to the project root.')
    parser.add_argument('--gene-gff', default='Sbicolor_454_v3.1.1.gene.gff3.gz')
    parser.add_argument('--exon-gff', default='Sbicolor_454_v3.1.1.gene_exons.gff3.gz')
    parser.add_argument('--locus-map', default='Sbicolor_454_v3.1.1.locus_transcript_name_map.txt')
    parser.add_argument('--annotation-info', default='Sbicolor_454_v3.1.1.P14.annotation_info.txt.gz')
    parser.add_argument('--defline', default='Sbicolor_454_v3.1.1.P14.defline.txt.gz')
    parser.add_argument('--synonym', default='Sbicolor_454_v3.1.1.synonym.txt')
    return parser.parse_args()


ARGS = parse_args()
BASE_DIR = os.path.abspath(ARGS.project_root)
OUT_DIR = os.path.join(BASE_DIR, ARGS.output_subdir)
os.makedirs(OUT_DIR, exist_ok=True)

# Raw phenotyping summaries
SIZE_SUMMARY = os.path.join(OUT_DIR, 'seed_size_summary.csv')
WEIGHT_SUMMARY = os.path.join(OUT_DIR, 'seed_weight_summary.csv')
COLOR_SUMMARY = os.path.join(OUT_DIR, 'color_summary.csv')
HYPER_AREA_SUMMARY = os.path.join(OUT_DIR, 'hyper_area_summary.csv')

# Spectral folders
F_DIR = os.path.join(BASE_DIR, ARGS.fluorescence_dir)
R_DIR = os.path.join(BASE_DIR, ARGS.reflectance_dir)

# HapMap genotype
HAPMAP_PATH = os.path.join(BASE_DIR, ARGS.hapmap_file)

# Sorghum gene annotation
GENE_GFF = os.path.join(BASE_DIR, ARGS.gene_gff)
EXON_GFF = os.path.join(BASE_DIR, ARGS.exon_gff)
LOCUS_MAP = os.path.join(BASE_DIR, ARGS.locus_map)
ANNOT_INFO = os.path.join(BASE_DIR, ARGS.annotation_info)
DEFLINE = os.path.join(BASE_DIR, ARGS.defline)
SYNONYM = os.path.join(BASE_DIR, ARGS.synonym)

# GEMMA/MLM outputs
MLM_DIR = os.path.join(OUT_DIR, 'mlm_gwas')
os.makedirs(MLM_DIR, exist_ok=True)

# Annotation outputs
ANNOT_DIR = os.path.join(OUT_DIR, 'annotation')
os.makedirs(ANNOT_DIR, exist_ok=True)

# =========================
# 1. BASIC HELPERS
# =========================

def read_csv_index(path, index_col="accession"):
    df = pd.read_csv(path)
    if index_col in df.columns:
        df = df.set_index(index_col)
    return df


def safe_mean(df, group_col="accession"):
    """Group by accession and take mean."""
    if group_col not in df.columns:
        return df
    return df.groupby(group_col).mean(numeric_only=True)


def build_spectral_matrix(folder, prefix, wl_min=None, wl_max=None):
    """
    Build accession x wavelength matrix from per-accession spectral files.
    Assumes each file has columns that can be parsed as numeric wavelengths.
    """
    all_rows = []
    accessions = []
    for fn in os.listdir(folder):
        if not (fn.endswith(".csv") or fn.endswith(".txt") or fn.endswith(".xlsx")):
            continue
        acc = os.path.splitext(fn)[0]
        path = os.path.join(folder, fn)
        try:
            df = pd.read_excel(path)
        except Exception:
            df = pd.read_csv(path)
        # pick numeric columns that look like wavelengths
        cols = []
        for c in df.columns:
            try:
                float(str(c).strip())
                cols.append(c)
            except Exception:
                continue
        if not cols:
            continue
        arr = df[cols].astype(float).mean(axis=0)
        row = pd.Series(arr.values, index=[f"{prefix}_{float(c):.1f}" for c in cols])
        all_rows.append(row)
        accessions.append(acc)
    if not all_rows:
        return pd.DataFrame()
    M = pd.DataFrame(all_rows)
    M.index = accessions
    # optional wavelength filtering
    keep = []
    for c in M.columns:
        try:
            wl = float(c.split("_", 1)[1])
        except Exception:
            wl = None
        if wl_min is not None and (wl is None or wl < wl_min):
            continue
        if wl_max is not None and (wl is None or wl > wl_max):
            continue
        keep.append(c)
    if keep:
        M = M[keep]
    return M.sort_index()


def compute_spectral_features(spec_df, prefix):
    """
    spec_df: accession x R_λ or F_λ.
    Returns feature DataFrame indexed by accession.
    """
    if spec_df.empty:
        return pd.DataFrame()

    feats = []
    for acc, row in spec_df.iterrows():
        vals = []
        wls = []
        for c, v in row.items():
            if not c.startswith(prefix + "_"):
                continue
            try:
                wl = float(c.split("_", 1)[1])
            except Exception:
                continue
            wls.append(wl)
            vals.append(float(v))
        if not wls:
            continue
        wls = np.array(wls)
        vals = np.array(vals)
        order = np.argsort(wls)
        wls = wls[order]
        vals = vals[order]

        # approximate wavelength spacing
        if len(wls) > 1:
            d = np.diff(wls)
            delta = float(np.median(d))
        else:
            delta = 1.0

        total = float(np.nansum(vals) * delta)
        peak_idx = int(np.nanargmax(vals))
        peak_wl = float(wls[peak_idx])
        peak_val = float(vals[peak_idx])

        # spectral centroid
        if np.nansum(vals) > 0:
            centroid = float(np.nansum(wls * vals) / np.nansum(vals))
        else:
            centroid = np.nan

        # entropy
        vpos = np.copy(vals)
        vpos[vpos < 0] = 0
        s = np.nansum(vpos)
        if s > 0:
            p = vpos / s
            p = p[p > 0]
            H = -np.nansum(p * np.log2(p))
            if len(p) > 1:
                H_norm = float(H / np.log2(len(p)))
            else:
                H_norm = 0.0
        else:
            H_norm = np.nan

        # band means
        def band_mean(lo, hi):
            mask = (wls >= lo) & (wls < hi)
            if not np.any(mask):
                return np.nan
            return float(np.nanmean(vals[mask]))

        feat = {
            "accession": acc,
            f"{prefix}_total": total,
            f"{prefix}_peak_wl": peak_wl,
            f"{prefix}_peak_val": peak_val,
            f"{prefix}_centroid": centroid,
            f"{prefix}_entropy": H_norm,
            f"{prefix}_blue_mean": band_mean(400, 500),
            f"{prefix}_green_mean": band_mean(500, 600),
            f"{prefix}_red_mean": band_mean(600, 700),
        }
        if prefix == "R":
            feat[f"{prefix}_nir_mean"] = band_mean(700, 900)
        feats.append(feat)

    feat_df = pd.DataFrame(feats).set_index("accession").sort_index()
    return feat_df


def compute_pca(df, n_components=5, prefix="PC"):
    """Simple PCA using SVD. No scaling, only mean-centering."""
    X = df.values.astype(float)
    mask = ~np.isnan(X).any(axis=1)
    X0 = X[mask]
    if X0.shape[0] < n_components + 1:
        raise RuntimeError("Not enough rows for PCA.")
    mean = X0.mean(axis=0, keepdims=True)
    Xc = X0 - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]
    scores = (Xc @ comps.T)
    pc_cols = [f"{prefix}{i+1}" for i in range(n_components)]
    res = pd.DataFrame(scores, index=df.index[mask], columns=pc_cols)
    return res


def standardize(df):
    return (df - df.mean()) / df.std(ddof=0)


# =========================
# 2. BUILD PHENOME + SPECTRA
# =========================

def build_phenome_and_spectra():
    print("[Step] Load phenotypic summaries")
    size = read_csv_index(SIZE_SUMMARY)
    weight = read_csv_index(WEIGHT_SUMMARY)
    color = read_csv_index(COLOR_SUMMARY)
    hyper = read_csv_index(HYPER_AREA_SUMMARY)

    # group by accession if needed
    size = safe_mean(size)
    weight = safe_mean(weight)
    color = safe_mean(color)
    hyper = safe_mean(hyper)

    # Merge phenotypes
    all_idx = set(size.index) | set(weight.index) | set(color.index) | set(hyper.index)
    all_idx = sorted(all_idx)
    pheno = pd.DataFrame(index=all_idx)
    for df in [size, weight, color, hyper]:
        pheno = pheno.join(df, how="left")

    # Build spectral matrices
    print("[Step] Build F and R spectral matrices")
    F_spec = build_spectral_matrix(F_DIR, prefix="F", wl_min=400, wl_max=750)
    R_spec = build_spectral_matrix(R_DIR, prefix="R", wl_min=None, wl_max=None)

    F_spec.to_csv(os.path.join(OUT_DIR, "hyperspec_F_full_matrix.csv"))
    R_spec.to_csv(os.path.join(OUT_DIR, "hyperspec_R_full_matrix.csv"))

    # Spectral features
    print("[Step] Spectral features and PCs")
    F_feat = compute_spectral_features(F_spec, prefix="F")
    R_feat = compute_spectral_features(R_spec, prefix="R")

    F_feat.to_csv(os.path.join(OUT_DIR, "hyperspec_F_features.csv"))
    R_feat.to_csv(os.path.join(OUT_DIR, "hyperspec_R_features.csv"))

    F_pcs = compute_pca(F_spec, n_components=5, prefix="F_PCPC")
    R_pcs = compute_pca(R_spec, n_components=5, prefix="R_PCPC")

    F_pcs.to_csv(os.path.join(OUT_DIR, "hyperspec_F_PCs.csv"))
    R_pcs.to_csv(os.path.join(OUT_DIR, "hyperspec_R_PCs.csv"))

    # Merge all into master
    master_idx = sorted(set(pheno.index) | set(F_feat.index) |
                        set(R_feat.index) | set(F_pcs.index) | set(R_pcs.index))
    master = pd.DataFrame(index=master_idx)
    for df in [pheno, F_feat, R_feat, F_pcs, R_pcs]:
        master = master.join(df, how="left")

    # Derived totals per area
    if "area_mean" in master.columns:
        if "R_total" in master.columns:
            master["R_total_per_area"] = master["R_total"] / master["area_mean"]
        if "F_total" in master.columns:
            master["F_total_per_area"] = master["F_total"] / master["area_mean"]

    master.to_csv(os.path.join(OUT_DIR, "seed_phenome_master_v5.csv"))
    print("  -> seed_phenome_master_v5.csv saved")

    return master, F_spec, R_spec


# =========================
# 3. GENOTYPE MATRIX + PCs
# =========================

def parse_hapmap_to_geno(hmp_path):
    """
    Parse HapMap to SNP x accession numeric matrix and SNP info.
    """
    print("[Step] Parse HapMap:", hmp_path)
    with open(hmp_path, "r") as f:
        header = f.readline().strip().split("\t")
    # fixed columns 0-10, then samples
    sample_ids = header[11:]

    rows = []
    info_records = []
    with open(hmp_path, "r") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 11:
                continue
            snp_id = parts[0]
            chrom = parts[2]
            pos = int(parts[3])
            alleles = parts[1]
            geno_strs = parts[11:]

            # count alleles
            alleles_clean = [a for a in alleles.split("/") if a not in ("N", "")]
            if len(alleles_clean) != 2:
                continue
            a1, a2 = alleles_clean
            # we call minor allele later
            g = []
            for gstr in geno_strs:
                if gstr in ("NN", "00", "--", ""):
                    g.append(np.nan)
                    continue
                if len(gstr) != 2:
                    g.append(np.nan)
                    continue
                x, y = gstr[0], gstr[1]
                if x not in (a1, a2) or y not in (a1, a2):
                    g.append(np.nan)
                    continue
                # dosage of second allele
                dosage = (x == a2) + (y == a2)
                g.append(float(dosage))
            g = np.array(g, dtype=float)
            # compute MAF and missingness
            valid = ~np.isnan(g)
            if valid.sum() == 0:
                continue
            p = np.nanmean(g[valid] / 2.0)
            maf = min(p, 1 - p)
            missing = 1.0 - valid.mean()
            if maf < 0.05 or missing > 0.20:
                continue

            rows.append((snp_id, g))
            info_records.append((snp_id, chrom, pos))

    if not rows:
        raise RuntimeError("No SNPs passed QC.")
    snp_ids = [r[0] for r in rows]
    mat = np.vstack([r[1] for r in rows])
    geno = pd.DataFrame(mat, index=snp_ids, columns=sample_ids)
    geno = geno.T  # accession x SNP
    snp_info = pd.DataFrame(info_records, columns=["snp", "chrom", "pos"]).set_index("snp")
    return geno, snp_info


def build_geno_and_pcs():
    geno, snp_info = parse_hapmap_to_geno(HAPMAP_PATH)
    geno.to_csv(os.path.join(OUT_DIR, "geno_matrix_filtered.csv"))
    snp_info.to_csv(os.path.join(OUT_DIR, "snp_info_filtered.csv"))
    print("  -> geno_matrix_filtered.csv shape:", geno.shape)

    # genotype PCs
    mask = ~np.isnan(geno.values).any(axis=1)
    X = geno.values[mask].astype(float)
    Xc = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=10)
    pcs = pca.fit_transform(Xc)
    pc_cols = [f"G_PCPC{i+1}" for i in range(pcs.shape[1])]
    geno_pcs = pd.DataFrame(pcs, index=geno.index[mask], columns=pc_cols)
    geno_pcs.to_csv(os.path.join(OUT_DIR, "geno_PCs.csv"))
    print("  -> geno_PCs.csv saved")
    return geno, snp_info, geno_pcs


# =========================
# 4. SPECTRAL HERITABILITY + MANTEL
# =========================

def multi_regression_R2(y, X):
    mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    if mask.sum() < X.shape[1] + 2:
        return np.nan
    y0 = y[mask]
    X0 = X[mask]
    X1 = np.column_stack([np.ones(X0.shape[0]), X0])
    XtX = X1.T @ X1
    try:
        beta = np.linalg.solve(XtX, X1.T @ y0)
    except np.linalg.LinAlgError:
        return np.nan
    y_hat = X1 @ beta
    ss_tot = np.sum((y0 - y0.mean())**2)
    ss_res = np.sum((y0 - y_hat)**2)
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def compute_heritability_spectrum(spec_df, geno_pcs, prefix, out_csv):
    if spec_df.empty or geno_pcs.empty:
        print("compute_heritability_spectrum: missing spec or geno_pcs.")
        return
    common = spec_df.index.intersection(geno_pcs.index)
    if len(common) < 20:
        print("Not enough overlapping accessions for heritability spectrum.")
        return
    S = spec_df.loc[common]
    X = geno_pcs.loc[common].values.astype(float)

    records = []
    for c in S.columns:
        if not c.startswith(prefix + "_"):
            continue
        y = S[c].values.astype(float)
        r2 = multi_regression_R2(y, X)
        records.append({"trait": c, "R2": r2})
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print("  -> heritability spectrum saved to", out_csv)


def mantel_test(D1, D2, n_perm=10000, random_state=42):
    rng = np.random.default_rng(random_state)
    # flatten upper triangle
    idx = np.triu_indices_from(D1, k=1)
    x = D1[idx]
    y = D2[idx]
    r_obs = np.corrcoef(x, y)[0, 1]
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(D2.shape[0])
        y_perm = D2[perm][:, perm][idx]
        r = np.corrcoef(x, y_perm)[0, 1]
        if r >= r_obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return r_obs, p


def compute_genetic_spectral_distance(geno_pcs, R_pcs):
    common = geno_pcs.index.intersection(R_pcs.index)
    G = geno_pcs.loc[common, ["G_PCPC1", "G_PCPC2", "G_PCPC3"]].values
    R = R_pcs.loc[common, ["R_PCPC1", "R_PCPC2", "R_PCPC3"]].values
    n = len(common)
    Dg = np.zeros((n, n))
    Dr = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dg = np.linalg.norm(G[i] - G[j])
            dr = np.linalg.norm(R[i] - R[j])
            Dg[i, j] = Dg[j, i] = dg
            Dr[i, j] = Dr[j, i] = dr
    r, p = mantel_test(Dg, Dr, n_perm=10000)
    out = pd.DataFrame({"r_mantel": [r], "p_value": [p]})
    out.to_csv(os.path.join(OUT_DIR, "genetic_spectral_mantel.csv"), index=False)
    print("  -> Mantel r =", r, "p =", p)


# =========================
# 5. GWAS-LITE + PC-CORRECTED GWAS
# =========================

def gwas_corr(geno_df, y):
    """
    Simple correlation-based GWAS (no covariates).
    geno_df: accession x SNP (0/1/2)
    y: pandas Series indexed by accession
    """
    common = geno_df.index.intersection(y.index)
    G = geno_df.loc[common]
    v = y.loc[common].values.astype(float)
    v = v - np.nanmean(v)
    res = []
    for snp in G.columns:
        g = G[snp].values.astype(float)
        mask = ~np.isnan(g) & ~np.isnan(v)
        if mask.sum() < 20:
            continue
        gv = g[mask] - np.nanmean(g[mask])
        vv = v[mask]
        num = np.sum(gv * vv)
        den = np.sqrt(np.sum(gv**2) * np.sum(vv**2))
        if den == 0:
            continue
        r = num / den
        n = mask.sum()
        # approximate t-test
        from math import sqrt, log
        t = r * sqrt((n-2)/(1-r**2+1e-12))
        # two-sided p using normal approx
        # (for large n; fine for screening)
        p = 2 * (1 - 0.5 * (1 + np.math.erf(abs(t)/np.sqrt(2))))
        res.append({"snp": snp, "r": r, "n": n, "p": p})
    return pd.DataFrame(res)


def gwas_with_covariates(geno_df, y, cov_df):
    """
    PC-corrected GWAS via residualization:
      y_res ~ g_res for each SNP
    """
    common = geno_df.index.intersection(y.index).intersection(cov_df.index)
    G = geno_df.loc[common]
    v = y.loc[common].values.astype(float)
    Xcov = cov_df.loc[common].values.astype(float)

    # regress y on covariates
    X1 = np.column_stack([np.ones(Xcov.shape[0]), Xcov])
    beta_y, *_ = np.linalg.lstsq(X1, v, rcond=None)
    y_hat = X1 @ beta_y
    y_res = v - y_hat

    res = []
    for snp in G.columns:
        g = G[snp].values.astype(float)
        mask = ~np.isnan(g)
        if mask.sum() < 20:
            continue
        g0 = g[mask]
        Xg = Xcov[mask]
        Xg1 = np.column_stack([np.ones(Xg.shape[0]), Xg])
        beta_g, *_ = np.linalg.lstsq(Xg1, g0, rcond=None)
        g_hat = Xg1 @ beta_g
        g_res = g0 - g_hat

        yy = y_res[mask]
        gv = g_res
        yy = yy - yy.mean()
        gv = gv - gv.mean()
        num = np.sum(yy * gv)
        den = np.sqrt(np.sum(yy**2) * np.sum(gv**2))
        if den == 0:
            continue
        r = num / den
        n = mask.sum()
        t = r * np.sqrt((n-2)/(1-r**2+1e-12))
        p = 2 * (1 - 0.5 * (1 + np.math.erf(abs(t)/np.sqrt(2))))
        res.append({"snp": snp, "r": r, "n": n, "p": p})
    return pd.DataFrame(res)


# =========================
# 6. GEMMA INPUT + QTL SUMMARY
# =========================

def prepare_gemma_inputs(master, R_spec, geno, snp_info):
    print("[Step] Preparing GEMMA input")
    # phenotypes: gray_mean, R_entropy from master
    traits = {}
    for col in ["gray_mean", "R_entropy"]:
        if col in master.columns:
            traits[col] = master[col].astype(float)

    # add R_650, R_748 from R_spec
    def pick_R_column(target_wl):
        cols = [c for c in R_spec.columns if c.startswith("R_")]
        wls = []
        for c in cols:
            try:
                wl = float(c.split("_", 1)[1])
            except Exception:
                continue
            wls.append((c, wl))
        arr = np.array([w for _, w in wls], dtype=float)
        idx = np.argmin(np.abs(arr - target_wl))
        return wls[idx]

    col650, real650 = pick_R_column(650.5)
    col748, real748 = pick_R_column(748.6)
    traits["R_650"] = R_spec[col650].astype(float)
    traits["R_748"] = R_spec[col748].astype(float)

    pheno_df = pd.DataFrame(traits)
    # restrict to accessions with genotype
    common = geno.index.intersection(pheno_df.index)
    pheno_df = pheno_df.loc[common].copy()
    pheno_df = pheno_df.fillna(-9)
    pheno_path = os.path.join(MLM_DIR, "gemma_pheno.txt")
    pheno_df.to_csv(pheno_path, sep="\t", header=False, index=False, float_format="%.6f")

    # write sample order
    sample_order = pd.Series(pheno_df.index, name="accession")
    sample_order.to_csv(os.path.join(MLM_DIR, "gemma_samples.txt"),
                        index=False, header=False)

    # genotype for GEMMA: SNP x accession, with fake alleles
    geno_for = geno.loc[common]
    geno_T = geno_for.T
    snp_ids = geno_T.index
    # create dummy allele columns
    geno_T.insert(0, "major", "G")
    geno_T.insert(0, "minor", "A")
    geno_T.insert(0, "snp", snp_ids)
    geno_path = os.path.join(MLM_DIR, "gemma_geno_fixed.txt")
    # lineterminator for Unix
    geno_T.to_csv(geno_path, sep="\t", header=False, index=False, lineterminator="\n")

    # annotation file
    anno = snp_info.reindex(snp_ids)
    anno_out = pd.DataFrame({
        "snp": snp_ids.astype(str),
        "chr": anno["chrom"].astype(str).values,
        "pos": anno["pos"].astype(float).values,
    })
    anno_path = os.path.join(MLM_DIR, "gemma_anno.txt")
    anno_out.to_csv(anno_path, sep="\t", header=False, index=False)

    print("  -> GEMMA files saved in", MLM_DIR)
    print("     gemma_pheno.txt, gemma_geno_fixed.txt, gemma_anno.txt")


def summarize_gemma_qtls():
    print("[Step] Summarizing GEMMA QTLs")
    gemma_out = os.path.join(MLM_DIR, "output")
    coloc_dir = os.path.join(MLM_DIR, "colocalization_gemma")
    os.makedirs(coloc_dir, exist_ok=True)

    trait_files = {
        "gray_mean": "senegal_gray_mean_lmm.assoc.txt",
        "R_650": "senegal_R650_lmm.assoc.txt",
        "R_748": "senegal_R748_lmm.assoc.txt",
        "R_entropy": "senegal_Rentropy_lmm.assoc.txt",
    }

    def load_assoc(path):
        if not os.path.exists(path):
            print("  !! missing:", path)
            return pd.DataFrame()
        df = pd.read_csv(path, sep=r"\s+")
        return df

    def call_qtls(df, trait, p_thresh=1e-5, window_kb=250.0):
        if df.empty:
            return pd.DataFrame()
        hits = df[df["p_wald"] < p_thresh].copy()
        if hits.empty:
            return pd.DataFrame()
        hits["pos"] = hits["ps"].astype(float)
        hits["chr"] = hits["chr"].astype(str)
        hits = hits.sort_values(["chr", "pos"])
        recs = []
        locus_id = 0
        for chrom in hits["chr"].unique():
            sub = hits[hits["chr"] == chrom].reset_index(drop=True)
            if sub.empty:
                continue
            start_pos = sub.loc[0, "pos"]
            locus_snps = [sub.loc[0]]
            for i in range(1, sub.shape[0]):
                row = sub.loc[i]
                if row["pos"] - start_pos <= window_kb * 1000.0:
                    locus_snps.append(row)
                else:
                    locus_id += 1
                    locus_df = pd.DataFrame(locus_snps)
                    lead = locus_df.loc[locus_df["p_wald"].idxmin()]
                    recs.append({
                        "trait": trait,
                        "locus_id": f"{chrom}_{locus_id}",
                        "chr": chrom,
                        "start": float(locus_df["pos"].min()),
                        "end": float(locus_df["pos"].max()),
                        "lead_pos": float(lead["pos"]),
                        "lead_p": float(lead["p_wald"]),
                        "n_snps": int(locus_df.shape[0]),
                        "lead_snp": lead.get("rs", np.nan),
                    })
                    start_pos = row["pos"]
                    locus_snps = [row]
            # last locus
            locus_id += 1
            locus_df = pd.DataFrame(locus_snps)
            lead = locus_df.loc[locus_df["p_wald"].idxmin()]
            recs.append({
                "trait": trait,
                "locus_id": f"{chrom}_{locus_id}",
                "chr": chrom,
                "start": float(locus_df["pos"].min()),
                "end": float(locus_df["pos"].max()),
                "lead_pos": float(lead["pos"]),
                "lead_p": float(lead["p_wald"]),
                "n_snps": int(locus_df.shape[0]),
                "lead_snp": lead.get("rs", np.nan),
            })
        return pd.DataFrame(recs)

    all_qtls = []
    for trait, fn in trait_files.items():
        path = os.path.join(gemma_out, fn)
        df = load_assoc(path)
        qtl = call_qtls(df, trait)
        qtl.to_csv(os.path.join(coloc_dir, f"QTL_{trait}_GEMMA.csv"), index=False)
        all_qtls.append(qtl)

    if not all_qtls:
        print("  !! No QTLs found.")
        return

    all_df = pd.concat(all_qtls, ignore_index=True)
    all_df.to_csv(os.path.join(coloc_dir, "QTL_all_traits_GEMMA.csv"), index=False)

    # colocalization across traits
    recs = []
    for i in range(all_df.shape[0]):
        for j in range(i+1, all_df.shape[0]):
            r1 = all_df.iloc[i]
            r2 = all_df.iloc[j]
            if r1["trait"] == r2["trait"]:
                continue
            if r1["chr"] != r2["chr"]:
                continue
            mid1 = 0.5 * (r1["start"] + r1["end"])
            mid2 = 0.5 * (r2["start"] + r2["end"])
            dist = abs(mid1 - mid2)
            if dist <= 250000.0:
                recs.append({
                    "trait1": r1["trait"],
                    "trait2": r2["trait"],
                    "chr": r1["chr"],
                    "pos1": mid1,
                    "pos2": mid2,
                    "dist_bp": dist,
                    "lead_snp1": r1["lead_snp"],
                    "lead_snp2": r2["lead_snp"],
                    "lead_p1": r1["lead_p"],
                    "lead_p2": r2["lead_p"],
                })
    coloc = pd.DataFrame(recs)
    coloc.to_csv(os.path.join(coloc_dir, "QTL_colocalization_pairs_GEMMA.csv"),
                 index=False)
    print("  -> GEMMA QTL and colocalization summary saved")


# =========================
# 7. PATH ANALYSIS
# =========================

def run_path_analysis(master, geno_pcs):
    print("[Step] Path analysis (G PCs -> morphology -> spectra)")
    # choose variables
    needed = ["seed_weight_mean", "area_mean", "R_nir_mean", "R_total"]
    for v in needed:
        if v not in master.columns:
            print("  !! missing variable:", v)
    # common accessions
    common = master.index.intersection(geno_pcs.index)
    df = master.loc[common, needed].copy()
    G = geno_pcs.loc[common, ["G_PCPC1", "G_PCPC2", "G_PCPC3"]].copy()

    # z-score
    ZG = standardize(G)
    ZM = standardize(df[["seed_weight_mean", "area_mean"]])
    ZS = standardize(df[["R_nir_mean", "R_total"]])

    def fit_ols(y, X):
        mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        y0 = y[mask]
        X0 = X[mask]
        X1 = np.column_stack([np.ones(X0.shape[0]), X0])
        beta, *_ = np.linalg.lstsq(X1, y0, rcond=None)
        return beta[1:]  # standardized so intercept ~0

    # G -> morphology
    recs = []
    for m_name in ZM.columns:
        beta = fit_ols(ZM[m_name].values, ZG.values)
        for k, pc_name in enumerate(ZG.columns):
            recs.append({
                "stage": "G_to_M",
                "response": m_name,
                "predictor": pc_name,
                "beta": beta[k],
            })

    # morphology + G -> spectra
    for s_name in ZS.columns:
        X = np.column_stack([ZG.values, ZM.values])
        beta = fit_ols(ZS[s_name].values, X)
        # first G PCs
        for k, pc_name in enumerate(ZG.columns):
            recs.append({
                "stage": "G+M_to_S",
                "response": s_name,
                "predictor": pc_name,
                "beta": beta[k],
            })
        # then morphology
        for j, m_name in enumerate(ZM.columns):
            recs.append({
                "stage": "G+M_to_S",
                "response": s_name,
                "predictor": m_name,
                "beta": beta[len(ZG.columns)+j],
            })

    coef_df = pd.DataFrame(recs)
    coef_df.to_csv(os.path.join(OUT_DIR, "path_regression_results.csv"),
                   index=False)

    # compute indirect and total effects
    # Indirect: G -> M -> S
    indir_recs = []
    pcs = list(ZG.columns)
    morphs = list(ZM.columns)
    specs = list(ZS.columns)

    def get_beta(stage, resp, pred):
        m = coef_df[(coef_df["stage"] == stage) &
                    (coef_df["response"] == resp) &
                    (coef_df["predictor"] == pred)]
        if m.empty:
            return 0.0
        return m["beta"].values[0]

    for pc in pcs:
        for s in specs:
            indirect = 0.0
            for m in morphs:
                a = get_beta("G_to_M", m, pc)
                b = get_beta("G+M_to_S", s, m)
                indirect += a * b
            direct = get_beta("G+M_to_S", s, pc)
            total = direct + indirect
            indir_recs.append({
                "genotype_PC": pc,
                "spectral_trait": s,
                "indirect_via_morph": indirect,
                "direct": direct,
                "total": total,
            })
    indir_df = pd.DataFrame(indir_recs)
    indir_df.to_csv(os.path.join(OUT_DIR, "path_indirect_effects.csv"),
                    index=False)
    print("  -> path_regression_results.csv and path_indirect_effects.csv saved")


# =========================
# 8. IN-SILICO SIMULATION
# =========================

def run_in_silico_simulation(master, geno_pcs):
    print("[Step] In-silico spectral manifold simulation")
    # use reflectance PCs
    R_pcs_cols = [c for c in master.columns if c.startswith("R_PCPC")]
    R_pcs = master[R_pcs_cols].copy()
    # common accessions
    common = master.index.intersection(geno_pcs.index).intersection(R_pcs.index)
    G = geno_pcs.loc[common]
    R = R_pcs.loc[common]

    Xg = G.values.astype(float)
    Yr = R.values.astype(float)
    reg = LinearRegression().fit(Xg, Yr)
    coef = reg.coef_.T
    intercept = reg.intercept_
    rpc_cols = list(R.columns)

    def predict_Rpcs(G_new):
        arr = intercept + G_new.values @ coef
        return pd.DataFrame(arr, index=G_new.index, columns=rpc_cols)

    # function to compute 2D manifold from R PCs
    def compute_manifold(df):
        arr = df.values.astype(float)
        arr_z = StandardScaler().fit_transform(arr)
        if HAS_UMAP:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            emb = reducer.fit_transform(arr_z)
        else:
            pca = PCA(n_components=2)
            emb = pca.fit_transform(arr_z)
        return pd.DataFrame(emb, index=df.index, columns=["dim1", "dim2"])

    # original manifold
    manifold_orig = compute_manifold(R)
    manifold_orig.to_csv(os.path.join(OUT_DIR, "manifold_original_RPC.csv"))

    # simulate KO/OE on first two PCs
    gpc_cols = list(G.columns)
    if len(gpc_cols) < 2:
        print("  !! need at least 2 genotype PCs")
        return
    pc1 = gpc_cols[0]
    pc2 = gpc_cols[1]
    experiments = {
        f"KO_{pc1}": (pc1, "KO"),
        f"OE_{pc1}": (pc1, "OE"),
        f"KO_{pc2}": (pc2, "KO"),
        f"OE_{pc2}": (pc2, "OE"),
    }

    for name, (pc, mode) in experiments.items():
        G_mod = G.copy()
        shift = 2.0 * G_mod[pc].std()
        if mode == "KO":
            G_mod[pc] = G_mod[pc] - shift
        else:
            G_mod[pc] = G_mod[pc] + shift
        R_mod = predict_Rpcs(G_mod)
        manifold = compute_manifold(R_mod)
        manifold.to_csv(os.path.join(OUT_DIR, f"manifold_{name}.csv"))

    print("  -> in-silico manifold CSVs saved")


# =========================
# 9. GENE ANNOTATION FOR GWAS HITS
# =========================

def load_gene_table():
    print("[Step] Load sorghum gene annotation")

    # basic gene coordinates
    with gzip.open(GENE_GFF, "rt") as f:
        rows = []
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            chrom, src, feat_type, start, end, score, strand, phase, attr = parts
            if feat_type != "gene":
                continue
            info = {}
            for a in attr.split(";"):
                if "=" in a:
                    k, v = a.split("=", 1)
                    info[k] = v
            gene_id = info.get("ID", "")
            rows.append((gene_id, chrom, int(start), int(end)))
    gene_df = pd.DataFrame(rows, columns=["gene_id", "chrom", "start", "end"])

    # locus map
    locus_map = pd.read_csv(LOCUS_MAP, sep="\t", header=None,
                            names=["locus", "transcript", "name"])
    # synonym
    syn_df = pd.read_csv(SYNONYM, sep="\t", header=None,
                         names=["locus", "synonym"])
    # annotation info
    with gzip.open(ANNOT_INFO, "rt") as f:
        ann_info = pd.read_csv(f, sep="\t", header=None)
    ann_info = ann_info.rename(columns={0: "locus", 1: "annotation_info"})

    # defline
    with gzip.open(DEFLINE, "rt") as f:
        def_df = pd.read_csv(f, sep="\t", header=None)
    def_df = def_df.rename(columns={0: "locus", 1: "defline"})

    # merge
    gene_df["locus"] = gene_df["gene_id"].str.replace("gene:", "", regex=False)
    g = gene_df.merge(locus_map, on="locus", how="left") \
               .merge(syn_df, on="locus", how="left") \
               .merge(ann_info, on="locus", how="left") \
               .merge(def_df, on="locus", how="left")
    g = g.rename(columns={"name": "gene_name"})
    g.to_csv(os.path.join(ANNOT_DIR, "gene_table_annotated.csv"), index=False)
    return g


def annotate_gwas_table(gwas_path, snp_info, gene_table,
                        out_path, logp_thresh=4.0,
                        gene_window_bp=50000):
    df = pd.read_csv(gwas_path)
    if "p" in df.columns:
        df["logp"] = -np.log10(df["p"])
    elif "p_wald" in df.columns:
        df["logp"] = -np.log10(df["p_wald"])
    else:
        return
    df = df.merge(snp_info.reset_index(), left_on="snp", right_on="snp", how="left")
    hits = df[df["logp"] >= logp_thresh].copy()
    if hits.empty:
        hits.to_csv(out_path, index=False)
        return

    # build simple chromosome index
    gene_by_chr = {str(ch): sub for ch, sub in gene_table.groupby("chrom")}

    records = []
    for _, row in hits.iterrows():
        chrom = str(row["chrom"])
        pos = float(row["pos"])
        genes = gene_by_chr.get(chrom)
        best_gene = None
        best_dist = None
        if genes is not None:
            dists = np.minimum(np.abs(pos - genes["start"]),
                               np.abs(pos - genes["end"]))
            idx = int(np.argmin(dists.values))
            dist = float(dists.iloc[idx])
            if dist <= gene_window_bp:
                best_gene = genes.iloc[idx]
                best_dist = dist
        rec = row.to_dict()
        if best_gene is not None:
            rec.update({
                "nearest_gene": best_gene["gene_id"],
                "gene_locus": best_gene["locus"],
                "gene_name": best_gene.get("gene_name", ""),
                "gene_dist_bp": best_dist,
                "gene_defline": best_gene.get("defline", ""),
                "gene_annotation": best_gene.get("annotation_info", "")
            })
        records.append(rec)
    ann = pd.DataFrame(records)
    ann.to_csv(out_path, index=False)


def summarize_gene_hits(annot_paths):
    all_hits = []
    for p in annot_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["source"] = os.path.basename(p)
            all_hits.append(df)
    if not all_hits:
        return
    all_df = pd.concat(all_hits, ignore_index=True)
    all_df.to_csv(os.path.join(ANNOT_DIR, "annotated_gwas_all_traits.csv"),
                  index=False)

    if "nearest_gene" not in all_df.columns:
        return
    gb = all_df.groupby("nearest_gene")
    recs = []
    for gene, sub in gb:
        traits = sorted(set(sub["source"]))
        recs.append({
            "nearest_gene": gene,
            "n_hits": sub.shape[0],
            "n_traits": len(traits),
            "sources": ";".join(traits),
            "min_p": sub["p"].min() if "p" in sub.columns else np.nan,
        })
    summary = pd.DataFrame(recs)
    summary.to_csv(os.path.join(ANNOT_DIR, "annotated_gwas_gene_summary.csv"),
                   index=False)


# =========================
# 10. MAIN ENTRY POINT
# =========================

def main():
    # phenome + spectra
    master, F_spec, R_spec = build_phenome_and_spectra()

    # genotype + PCs
    geno, snp_info, geno_pcs = build_geno_and_pcs()

    # overlap set
    common = master.index.intersection(geno.index).intersection(R_spec.index)
    master = master.loc[common]
    F_spec = F_spec.loc[common]
    R_spec = R_spec.loc[common]
    geno = geno.loc[common]
    geno_pcs = geno_pcs.loc[common]

    master.to_csv(os.path.join(OUT_DIR, "seed_phenome_master_v5.csv"))

    # spectral heritability
    compute_heritability_spectrum(R_spec, geno_pcs,
                                  prefix="R",
                                  out_csv=os.path.join(OUT_DIR, "R_heritability_spectrum.csv"))
    compute_heritability_spectrum(F_spec, geno_pcs,
                                  prefix="F",
                                  out_csv=os.path.join(OUT_DIR, "F_heritability_spectrum.csv"))

    # genetic-spectral distance + Mantel
    R_pcs = master[[c for c in master.columns if c.startswith("R_PCPC")]]
    compute_genetic_spectral_distance(geno_pcs, R_pcs)

    # GWAS-lite for a few traits (just compute tables, no plots)
    traits = ["gray_mean", "R_total", "R_centroid", "R_entropy", "F_total"]
    for t in traits:
        if t not in master.columns:
            continue
        res = gwas_corr(geno, master[t])
        out = os.path.join(OUT_DIR, f"gwas_{t}.csv")
        res.to_csv(out, index=False)

    # PC-corrected GWAS for manifold and key wavelengths
    # Assume R_manifold_coords already computed in notebook or re-compute if needed
    # Here we only show how we would call it:
    for trait_name, col in [("R_UMAP1", "R_UMAP1"), ("R_UMAP2", "R_UMAP2")]:
        if col in master.columns:
            res = gwas_with_covariates(geno, master[col], geno_pcs[["G_PCPC1","G_PCPC2","G_PCPC3"]])
            res.to_csv(os.path.join(OUT_DIR, f"gwas_withPC_{trait_name}.csv"),
                       index=False)

    # GEMMA input and QTL summary (requires external GEMMA run)
    prepare_gemma_inputs(master, R_spec, geno, snp_info)
    # After you run GEMMA externally, call:
    # summarize_gemma_qtls()

    # Path analysis
    run_path_analysis(master, geno_pcs)

    # In-silico simulation
    run_in_silico_simulation(master, geno_pcs)

    # Gene annotation of GWAS-lite + PC-corrected results
    gene_table = load_gene_table()
    annot_paths = []
    # example: annotate gray_mean, R_650.5nm, R_748.6nm, etc. if available
    for fname in os.listdir(OUT_DIR):
        if fname.startswith("gwas_") and fname.endswith(".csv"):
            trait_name = fname.replace(".csv", "")
            in_path = os.path.join(OUT_DIR, fname)
            out_path = os.path.join(ANNOT_DIR, f"annotated_{trait_name}.csv")
            annotate_gwas_table(in_path, snp_info, gene_table, out_path)
            annot_paths.append(out_path)

    summarize_gene_hits(annot_paths)
    print("All core analyses completed.")


if __name__ == "__main__":
    main()
