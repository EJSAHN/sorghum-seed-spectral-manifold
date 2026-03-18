#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared utilities for Senegal sorghum hyperspectral support analyses.

Design goals
------------
- GitHub-ready: no hard-coded directories.
- Robust to messy project layouts: recursive discovery + optional CLI overrides.
- Compatible with either raw HapMap input or pre-filtered genotype matrices.
- Produces tables that support manuscript-level analyses and supplemental outputs.
"""

from __future__ import annotations

import json
import math
import os
import re
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXCLUDE_DIR_TOKENS = {
    ".git", "__pycache__", "node_modules", "env", "venv", ".venv",
    "revision_outputs", "revision_figures", "revision_tables", "dist", "build"
}

DEFAULT_MAIN_TRAIT_KEYWORDS = [
    "gray", "brightness", "entropy", "centroid", "650", "748", "750", "umap"
]

DEFAULT_HIGHLIGHT_GENES = ["Sobic.006G072601", "Y", "Tannin1"]


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "item"



def normalize_accession(x: object) -> str:
    s = str(x).strip()
    s = re.sub(r"\s+", "", s)
    return s



def normalize_chr(x: object) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    if m:
        return str(int(m.group(1)))
    return s if s else None



def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path



def safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")



def pick_first_existing(paths: Iterable[Optional[Path]]) -> Optional[Path]:
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None



def list_files_recursive(project_root: Path) -> List[Path]:
    out: List[Path] = []
    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_TOKENS and not d.startswith('.')]
        root_path = Path(root)
        out.extend(root_path / f for f in files)
    return out



def choose_best_path(paths: Sequence[Path], preferred_patterns: Sequence[str]) -> Optional[Path]:
    if not paths:
        return None

    def score(p: Path) -> Tuple[int, int, int, str]:
        name = p.name.lower()
        parts = {part.lower() for part in p.parts}
        pref = 0
        for i, pat in enumerate(preferred_patterns):
            if pat.lower() in name:
                pref += 1000 - i
        shallow = -len(p.parts)
        rootish = 100 if any(tok in parts for tok in {"paper1_output_spectral_v5", "_paper1_output_spectral_v5"}) else 0
        return (pref, rootish, shallow, str(p))

    return sorted(paths, key=score, reverse=True)[0]


# ---------------------------------------------------------------------------
# Asset discovery
# ---------------------------------------------------------------------------


@dataclass
class AssetDiscovery:
    project_root: Path
    phenotype_csv: Optional[Path]
    supplement_xlsx: Optional[Path]
    genotype_csv: Optional[Path]
    snp_info_csv: Optional[Path]
    hapmap_path: Optional[Path]
    gff_path: Optional[Path]
    synonym_path: Optional[Path]
    defline_path: Optional[Path]
    annotation_info_path: Optional[Path]
    assoc_files: List[Path]
    alternates: Dict[str, List[str]]

    def to_manifest_records(self) -> pd.DataFrame:
        rows = []
        selected = {
            "phenotype_csv": self.phenotype_csv,
            "supplement_xlsx": self.supplement_xlsx,
            "genotype_csv": self.genotype_csv,
            "snp_info_csv": self.snp_info_csv,
            "hapmap_path": self.hapmap_path,
            "gff_path": self.gff_path,
            "synonym_path": self.synonym_path,
            "defline_path": self.defline_path,
            "annotation_info_path": self.annotation_info_path,
            "assoc_files": "; ".join(str(p) for p in self.assoc_files) if self.assoc_files else "",
        }
        for key, value in selected.items():
            rows.append({
                "asset": key,
                "selected_path": str(value) if value else "",
                "alternate_candidates": " | ".join(self.alternates.get(key, [])),
            })
        return pd.DataFrame(rows)



def discover_assets(
    project_root: Path,
    phenotype_csv: Optional[Path] = None,
    supplement_xlsx: Optional[Path] = None,
    genotype_csv: Optional[Path] = None,
    snp_info_csv: Optional[Path] = None,
    hapmap_path: Optional[Path] = None,
    gff_path: Optional[Path] = None,
    synonym_path: Optional[Path] = None,
    defline_path: Optional[Path] = None,
    annotation_info_path: Optional[Path] = None,
    assoc_dir: Optional[Path] = None,
) -> AssetDiscovery:
    all_files = list_files_recursive(project_root)

    def find_candidates(predicate) -> List[Path]:
        return [p for p in all_files if predicate(p)]

    phenotype_candidates = find_candidates(
        lambda p: p.suffix.lower() == ".csv" and (
            "seed_phenome_master" in p.name.lower() or
            ("phenome" in p.name.lower() and "seed" in p.name.lower())
        )
    )
    supplement_candidates = find_candidates(
        lambda p: p.suffix.lower() in {".xlsx", ".xlsm", ".xls"} and "supplementary_data" in p.name.lower()
    )
    genotype_candidates = find_candidates(lambda p: p.name.lower() == "geno_matrix_filtered.csv")
    snp_info_candidates = find_candidates(lambda p: p.name.lower() == "snp_info_filtered.csv")
    hapmap_candidates = find_candidates(
        lambda p: any(tok in p.name.lower() for tok in [".hmp.txt", ".hapmap", ".hmp.txt.gz"]) or p.name.lower().endswith(".hmp")
    )
    gff_candidates = find_candidates(
        lambda p: p.name.lower().endswith((".gff3", ".gff3.gz", ".gff", ".gff.gz")) and "gene" in p.name.lower()
    )
    synonym_candidates = find_candidates(lambda p: "synonym" in p.name.lower() and p.suffix.lower() in {".txt", ".gz"})
    defline_candidates = find_candidates(lambda p: "defline" in p.name.lower() and p.suffix.lower() in {".txt", ".gz"})
    annot_info_candidates = find_candidates(lambda p: "annotation_info" in p.name.lower() and p.suffix.lower() in {".txt", ".gz"})

    if assoc_dir is not None and assoc_dir.exists():
        assoc_candidates = sorted([p for p in assoc_dir.rglob("*") if p.is_file() and p.name.lower().endswith((".assoc", ".assoc.txt"))])
    else:
        assoc_candidates = find_candidates(lambda p: p.name.lower().endswith((".assoc", ".assoc.txt")))

    alternates = {
        "phenotype_csv": [str(p) for p in phenotype_candidates],
        "supplement_xlsx": [str(p) for p in supplement_candidates],
        "genotype_csv": [str(p) for p in genotype_candidates],
        "snp_info_csv": [str(p) for p in snp_info_candidates],
        "hapmap_path": [str(p) for p in hapmap_candidates],
        "gff_path": [str(p) for p in gff_candidates],
        "synonym_path": [str(p) for p in synonym_candidates],
        "defline_path": [str(p) for p in defline_candidates],
        "annotation_info_path": [str(p) for p in annot_info_candidates],
        "assoc_files": [str(p) for p in assoc_candidates],
    }

    return AssetDiscovery(
        project_root=project_root,
        phenotype_csv=pick_first_existing([phenotype_csv]) or choose_best_path(phenotype_candidates, ["seed_phenome_master_v5", "seed_phenome_master"]),
        supplement_xlsx=pick_first_existing([supplement_xlsx]) or choose_best_path(supplement_candidates, ["supplementary_data_s1", "supplementary_data"]),
        genotype_csv=pick_first_existing([genotype_csv]) or choose_best_path(genotype_candidates, ["geno_matrix_filtered"]),
        snp_info_csv=pick_first_existing([snp_info_csv]) or choose_best_path(snp_info_candidates, ["snp_info_filtered"]),
        hapmap_path=pick_first_existing([hapmap_path]) or choose_best_path(hapmap_candidates, ["combined_anthracnosemarch", ".hmp.txt"]),
        gff_path=pick_first_existing([gff_path]) or choose_best_path(gff_candidates, ["gene.gff3", "gene.gff"]),
        synonym_path=pick_first_existing([synonym_path]) or choose_best_path(synonym_candidates, ["synonym"]),
        defline_path=pick_first_existing([defline_path]) or choose_best_path(defline_candidates, ["defline"]),
        annotation_info_path=pick_first_existing([annotation_info_path]) or choose_best_path(annot_info_candidates, ["annotation_info"]),
        assoc_files=sorted(assoc_candidates),
        alternates=alternates,
    )


# ---------------------------------------------------------------------------
# Phenotype loaders
# ---------------------------------------------------------------------------


def _set_best_index(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["accession", "Accessions", "Genotype", "Taxa", "taxa", "genotype"]:
        if col in df.columns:
            out = df.copy()
            out.index = out[col].map(normalize_accession)
            return out
    if df.index.name is None:
        out = df.copy()
        out.index = [normalize_accession(x) for x in out.index]
        return out
    out = df.copy()
    out.index = out.index.map(normalize_accession)
    return out



def load_phenotype_table(assets: AssetDiscovery) -> Tuple[pd.DataFrame, str]:
    if assets.phenotype_csv and assets.phenotype_csv.exists():
        df = pd.read_csv(assets.phenotype_csv)
        df = _set_best_index(df)
        return df, str(assets.phenotype_csv)

    if assets.supplement_xlsx and assets.supplement_xlsx.exists():
        xls = pd.ExcelFile(assets.supplement_xlsx)
        sheet = "Accessions_Traits" if "Accessions_Traits" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
        df = _set_best_index(df)
        return df, f"{assets.supplement_xlsx}::{sheet}"

    raise FileNotFoundError("Could not locate phenotype table (seed_phenome_master*.csv or Supplementary_Data_S1.xlsx).")



def find_trait_column(phenotype_df: pd.DataFrame, trait_name: str) -> Optional[str]:
    cols = list(phenotype_df.columns)
    lower_map = {c.lower(): c for c in cols}
    if trait_name in cols:
        return trait_name
    if trait_name.lower() in lower_map:
        return lower_map[trait_name.lower()]

    norm = trait_name.lower().replace(".", "").replace("nm", "").replace("_", "").replace("-", "")
    candidates = []
    for c in cols:
        c_norm = c.lower().replace(".", "").replace("nm", "").replace("_", "").replace("-", "")
        if c_norm == norm:
            return c
        if norm in c_norm or c_norm in norm:
            candidates.append(c)

    aliases = {
        "r748": ["R750_Mean", "R_748.6nm", "R750"],
        "r750": ["R750_Mean", "R_748.6nm", "R750"],
        "r650": ["R650_Mean", "R_650.5nm", "R650"],
        "gray": ["gray_mean"],
        "brightness": ["gray_mean"],
        "entropy": ["R_entropy"],
        "centroid": ["R_centroid"],
    }
    for key, opts in aliases.items():
        if key in norm:
            for opt in opts:
                if opt in cols:
                    return opt
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Genotype loaders
# ---------------------------------------------------------------------------


def load_genotype_and_snpinfo(assets: AssetDiscovery) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    if assets.genotype_csv and assets.snp_info_csv and assets.genotype_csv.exists() and assets.snp_info_csv.exists():
        geno = pd.read_csv(assets.genotype_csv, index_col=0)
        geno.index = geno.index.map(normalize_accession)
        snp_info = pd.read_csv(assets.snp_info_csv, index_col=0)
        snp_info.index = snp_info.index.astype(str)
        if "chrom" not in snp_info.columns or "pos" not in snp_info.columns:
            raise ValueError("snp_info_filtered.csv must contain chrom and pos columns.")
        snp_info["chrom"] = snp_info["chrom"].map(normalize_chr)
        snp_info["pos"] = pd.to_numeric(snp_info["pos"], errors="coerce")
        return geno, snp_info, f"{assets.genotype_csv} + {assets.snp_info_csv}"

    if assets.hapmap_path and assets.hapmap_path.exists():
        geno, snp_info = parse_hapmap_to_geno(assets.hapmap_path)
        return geno, snp_info, str(assets.hapmap_path)

    raise FileNotFoundError("Could not locate genotype input (geno_matrix_filtered.csv + snp_info_filtered.csv or HapMap).")



def open_textmaybe_gz(path: Path):
    return gzip.open(path, "rt") if str(path).lower().endswith(".gz") else open(path, "r")



def parse_hapmap_to_geno(hmp_path: Path, maf_min: float = 0.05, missing_max: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open_textmaybe_gz(hmp_path) as handle:
        header = handle.readline().strip().split("\t")
        sample_ids = [normalize_accession(x) for x in header[11:]]

    rows: List[np.ndarray] = []
    snp_records: List[Tuple[str, str, int]] = []

    with open_textmaybe_gz(hmp_path) as handle:
        next(handle)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12:
                continue
            snp = str(parts[0])
            alleles = str(parts[1])
            chrom = normalize_chr(parts[2])
            try:
                pos = int(float(parts[3]))
            except Exception:
                continue
            allele_list = [a for a in alleles.split("/") if a and a not in {"N", "-"}]
            if len(allele_list) != 2:
                continue
            a1, a2 = allele_list
            calls = []
            for gt in parts[11:]:
                gt = gt.strip().upper()
                if gt in {"NN", "N", "00", "--", "", "NA"}:
                    calls.append(np.nan)
                    continue
                if len(gt) == 1:
                    gt = gt * 2
                if len(gt) != 2:
                    calls.append(np.nan)
                    continue
                x, y = gt[0], gt[1]
                if x not in {a1, a2} or y not in {a1, a2}:
                    calls.append(np.nan)
                    continue
                dosage = float((x == a2) + (y == a2))
                calls.append(dosage)
            g = np.asarray(calls, dtype=np.float32)
            valid = np.isfinite(g)
            if valid.sum() == 0:
                continue
            p = float(np.nanmean(g[valid] / 2.0))
            maf = min(p, 1.0 - p)
            missing = float(1.0 - valid.mean())
            if maf < maf_min or missing > missing_max:
                continue
            rows.append(g)
            snp_records.append((snp, chrom, pos))

    if not rows:
        raise RuntimeError("No SNPs passed HapMap QC thresholds.")

    mat = np.vstack(rows)
    snp_ids = [r[0] for r in snp_records]
    geno = pd.DataFrame(mat, index=snp_ids, columns=sample_ids).T
    snp_info = pd.DataFrame(snp_records, columns=["snp", "chrom", "pos"]).set_index("snp")
    return geno, snp_info



def align_phenotype_and_genotype(phenotype_df: pd.DataFrame, geno_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ph = phenotype_df.copy()
    gn = geno_df.copy()
    ph.index = ph.index.map(normalize_accession)
    gn.index = gn.index.map(normalize_accession)
    common = ph.index.intersection(gn.index)
    if len(common) < 20:
        raise RuntimeError(f"Too few overlapping accessions after alignment: {len(common)}")
    return ph.loc[common].copy(), gn.loc[common].copy()



def mean_impute_and_standardize(geno_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    G = geno_df.to_numpy(dtype=np.float32, copy=True)
    means = np.nanmean(G, axis=0)
    inds = np.where(~np.isfinite(G))
    if inds[0].size:
        G[inds] = means[inds[1]]
    stds = G.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    Z = (G - means) / stds
    return G, means, Z


# ---------------------------------------------------------------------------
# Gene annotation loaders
# ---------------------------------------------------------------------------


def parse_gff_attributes(attr_str: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for item in str(attr_str).split(";"):
        if not item:
            continue
        if "=" in item:
            k, v = item.split("=", 1)
            attrs[k.strip()] = v.strip()
        elif " " in item:
            k, v = item.split(" ", 1)
            attrs[k.strip()] = v.strip().strip('"')
    return attrs



def load_gene_table(assets: AssetDiscovery) -> Tuple[pd.DataFrame, str]:
    if assets.gff_path and assets.gff_path.exists():
        gene_df = load_gene_table_from_gff(assets)
        return gene_df, str(assets.gff_path)

    if assets.supplement_xlsx and assets.supplement_xlsx.exists():
        gene_df = load_gene_table_from_supplement(assets.supplement_xlsx)
        return gene_df, f"{assets.supplement_xlsx}::CandidateGenes_mainTraits"

    raise FileNotFoundError("Could not locate a gene annotation source (GFF or Supplementary_Data_S1.xlsx).")



def load_gene_table_from_gff(assets: AssetDiscovery) -> pd.DataFrame:
    rows = []
    with open_textmaybe_gz(assets.gff_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, source, feature, start, end, score, strand, phase, attrs = parts
            if feature.lower() != "gene":
                continue
            attr = parse_gff_attributes(attrs)
            gene_id = attr.get("ID") or attr.get("gene_id") or attr.get("Name")
            name = attr.get("Name") or attr.get("gene_name") or gene_id
            if not gene_id:
                continue
            rows.append({
                "gene_id": gene_id,
                "chrom": normalize_chr(chrom),
                "start": int(start),
                "end": int(end),
                "gene_name": name,
            })
    gene = pd.DataFrame(rows).drop_duplicates(subset=["gene_id"]).reset_index(drop=True)
    if gene.empty:
        raise RuntimeError("Parsed GFF but recovered zero genes.")

    # Optional annotation text merge
    desc_map = {}
    if assets.annotation_info_path and assets.annotation_info_path.exists():
        try:
            with open_textmaybe_gz(assets.annotation_info_path) as handle:
                ann = pd.read_csv(handle, sep="\t", header=None, usecols=[0, 1], names=["gene_id", "annotation"])
            desc_map.update(dict(zip(ann["gene_id"].astype(str), ann["annotation"].astype(str))))
        except Exception:
            pass
    if assets.defline_path and assets.defline_path.exists():
        try:
            with open_textmaybe_gz(assets.defline_path) as handle:
                dfl = pd.read_csv(handle, sep="\t", header=None, usecols=[0, 1], names=["gene_id", "defline"])
            for k, v in zip(dfl["gene_id"].astype(str), dfl["defline"].astype(str)):
                desc_map.setdefault(k, v)
        except Exception:
            pass
    gene["description"] = gene["gene_id"].map(desc_map).fillna("")

    # Optional synonym merge
    gene["aliases"] = ""
    if assets.synonym_path and assets.synonym_path.exists():
        try:
            with open_textmaybe_gz(assets.synonym_path) as handle:
                syn = pd.read_csv(handle, sep="\t", header=None, low_memory=False)
            syn = syn.iloc[:, :2].copy()
            syn.columns = ["gene_id", "alias"]
            syn["gene_id"] = syn["gene_id"].astype(str)
            alias_map = syn.groupby("gene_id")["alias"].apply(lambda x: ";".join(sorted({str(v) for v in x if pd.notna(v)})))
            gene["aliases"] = gene["gene_id"].map(alias_map).fillna("")
        except Exception:
            pass
    return gene



def load_gene_table_from_supplement(supplement_xlsx: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(supplement_xlsx)
    sheets = [s for s in xls.sheet_names if s.lower().startswith("candidategenes")]
    if not sheets:
        raise RuntimeError("Supplementary workbook lacks CandidateGenes sheets.")
    frames = []
    for sheet in sheets:
        df = pd.read_excel(xls, sheet_name=sheet)
        col_map = {c.lower(): c for c in df.columns}
        id_col = col_map.get("gene id") or col_map.get("gene_id") or col_map.get("gene")
        chr_col = col_map.get("gene_chrom") or col_map.get("chr") or col_map.get("chrom") or col_map.get("gene chr")
        start_col = col_map.get("gene_start") or col_map.get("start")
        end_col = col_map.get("gene_end") or col_map.get("end")
        desc_col = col_map.get("sorghum_defline") or col_map.get("description") or col_map.get("desc")
        if not all([id_col, chr_col, start_col, end_col]):
            continue
        sub = pd.DataFrame({
            "gene_id": df[id_col].astype(str),
            "chrom": df[chr_col].map(normalize_chr),
            "start": pd.to_numeric(df[start_col], errors="coerce"),
            "end": pd.to_numeric(df[end_col], errors="coerce"),
            "gene_name": df[id_col].astype(str),
            "description": df[desc_col].astype(str) if desc_col else "",
        })
        frames.append(sub)
    gene = pd.concat(frames, ignore_index=True).dropna(subset=["chrom", "start", "end"]).drop_duplicates(subset=["gene_id"]).reset_index(drop=True)
    gene["start"] = gene["start"].astype(int)
    gene["end"] = gene["end"].astype(int)
    gene["aliases"] = ""
    return gene


# ---------------------------------------------------------------------------
# Association file utilities
# ---------------------------------------------------------------------------


def infer_trait_name_from_path(path: Path) -> str:
    name = path.name
    name = re.sub(r"\.assoc(?:\.txt)?$", "", name, flags=re.I)
    name = re.sub(r"_lmm$", "", name, flags=re.I)
    name = re.sub(r"^GWAS_", "", name, flags=re.I)
    name = re.sub(r"^senegal_", "", name, flags=re.I)
    return name



def load_assoc_table(path: Path, trait_name: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] <= 1:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    col_map = {c.lower(): c for c in df.columns}
    chr_col = col_map.get("chr") or col_map.get("chrom") or col_map.get("chromosome")
    pos_col = col_map.get("ps") or col_map.get("bp") or col_map.get("pos") or col_map.get("position")
    p_col = col_map.get("p_wald") or col_map.get("p_lrt") or col_map.get("p_score") or col_map.get("p") or col_map.get("pvalue")
    snp_col = col_map.get("rs") or col_map.get("snp") or col_map.get("marker")
    beta_col = col_map.get("beta") or col_map.get("effect")
    if chr_col is None or pos_col is None or p_col is None:
        raise ValueError(f"Association file missing chr/pos/p columns: {path}")

    # Some GEMMA exports in this project store physical position in `chr`
    # and chromosome number in `ps`, while others use the conventional layout.
    # Detect the swapped case heuristically so both formats load correctly.
    chr_vals = pd.to_numeric(df[chr_col], errors="coerce")
    pos_vals = pd.to_numeric(df[pos_col], errors="coerce")
    if chr_vals.notna().mean() > 0.9 and pos_vals.notna().mean() > 0.9:
        chr_small = float(chr_vals.between(1, 20).mean())
        pos_small = float(pos_vals.between(1, 20).mean())
        chr_large = float((chr_vals > 1000).mean())
        pos_large = float((pos_vals > 1000).mean())
        if chr_small < 0.2 and chr_large > 0.8 and pos_small > 0.8 and pos_large < 0.2:
            chr_col, pos_col = pos_col, chr_col

    out = pd.DataFrame({
        "trait": trait_name or infer_trait_name_from_path(path),
        "snp": df[snp_col].astype(str) if snp_col else [f"{normalize_chr(c)}_{int(float(p))}" for c, p in zip(df[chr_col], df[pos_col])],
        "chrom": [normalize_chr(x) for x in df[chr_col]],
        "pos": pd.to_numeric(df[pos_col], errors="coerce"),
        "p": pd.to_numeric(df[p_col], errors="coerce"),
    })
    if beta_col is not None:
        out["beta"] = pd.to_numeric(df[beta_col], errors="coerce")
    out = out.dropna(subset=["chrom", "pos", "p"]).sort_values(["chrom", "pos", "p"]).reset_index(drop=True)
    out["minus_log10p"] = -np.log10(out["p"].clip(lower=np.nextafter(0, 1)))
    return out



def load_assoc_tables_from_discovery(assets: AssetDiscovery) -> Dict[str, pd.DataFrame]:
    assoc_tables: Dict[str, pd.DataFrame] = {}
    for p in assets.assoc_files:
        trait = infer_trait_name_from_path(p)
        try:
            assoc_tables[trait] = load_assoc_table(p, trait_name=trait)
        except Exception:
            continue

    if assoc_tables:
        return assoc_tables

    # Fallback: use significant SNP sheet from supplement workbook
    if assets.supplement_xlsx and assets.supplement_xlsx.exists():
        xls = pd.ExcelFile(assets.supplement_xlsx)
        if "GWAS_SNPs_mainTraits" in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name="GWAS_SNPs_mainTraits")
            col_map = {c.lower(): c for c in df.columns}
            trait_col = col_map.get("trait")
            snp_col = col_map.get("snp")
            chrom_col = col_map.get("chrom")
            pos_col = col_map.get("pos")
            p_col = col_map.get("p")
            for trait, sub in df.groupby(df[trait_col].astype(str)):
                out = pd.DataFrame({
                    "trait": trait,
                    "snp": sub[snp_col].astype(str),
                    "chrom": sub[chrom_col].map(normalize_chr),
                    "pos": pd.to_numeric(sub[pos_col], errors="coerce"),
                    "p": pd.to_numeric(sub[p_col], errors="coerce"),
                }).dropna(subset=["chrom", "pos", "p"])
                out["minus_log10p"] = -np.log10(out["p"].clip(lower=np.nextafter(0, 1)))
                assoc_tables[trait] = out.sort_values(["chrom", "pos", "p"]).reset_index(drop=True)
    return assoc_tables


# ---------------------------------------------------------------------------
# LD utilities
# ---------------------------------------------------------------------------


def pairwise_r2(x: np.ndarray, y: np.ndarray, min_non_missing: int = 20) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_non_missing:
        return float("nan")
    xv = x[mask].astype(np.float32)
    yv = y[mask].astype(np.float32)
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    denom = float(np.sqrt((xv * xv).sum() * (yv * yv).sum()))
    if denom == 0:
        return float("nan")
    r = float((xv * yv).sum() / denom)
    return r * r



def compute_ld_decay(
    geno_df: pd.DataFrame,
    snp_info: pd.DataFrame,
    max_distance_bp: int = 2_000_000,
    distance_bin_bp: int = 25_000,
    max_snps_per_chr: int = 600,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    geno = geno_df.copy()
    snp_info = snp_info.copy()
    shared_snps = [s for s in snp_info.index if s in geno.columns]
    snp_info = snp_info.loc[shared_snps].copy()
    geno = geno[shared_snps].copy()

    records = []
    per_chr_summary = []
    for chrom in sorted(snp_info["chrom"].dropna().astype(str).unique(), key=lambda x: int(re.search(r"\d+", x).group(0)) if re.search(r"\d+", x) else x):
        sub_info = snp_info[snp_info["chrom"].astype(str) == str(chrom)].sort_values("pos")
        if sub_info.empty:
            continue
        snps = list(sub_info.index)
        if len(snps) > max_snps_per_chr:
            idx = np.sort(rng.choice(len(snps), size=max_snps_per_chr, replace=False))
            snps = [snps[i] for i in idx]
            sub_info = sub_info.loc[snps].sort_values("pos")
        positions = sub_info["pos"].astype(int).to_numpy()
        snps = list(sub_info.index)
        G = geno[snps].to_numpy(dtype=np.float32, copy=False)
        n_pairs = 0
        r2_values = []
        d_values = []
        for i in range(len(snps)):
            j = i + 1
            while j < len(snps) and (positions[j] - positions[i]) <= max_distance_bp:
                r2 = pairwise_r2(G[:, i], G[:, j])
                if np.isfinite(r2):
                    dist = int(positions[j] - positions[i])
                    records.append({
                        "chrom": str(chrom),
                        "snp1": snps[i],
                        "pos1": int(positions[i]),
                        "snp2": snps[j],
                        "pos2": int(positions[j]),
                        "distance_bp": dist,
                        "r2": r2,
                    })
                    n_pairs += 1
                    r2_values.append(r2)
                    d_values.append(dist)
                j += 1
        per_chr_summary.append({
            "chrom": str(chrom),
            "n_snps_sampled": len(snps),
            "n_pairs_tested": n_pairs,
            "mean_r2": float(np.nanmean(r2_values)) if r2_values else np.nan,
            "median_r2": float(np.nanmedian(r2_values)) if r2_values else np.nan,
            "median_distance_bp": float(np.nanmedian(d_values)) if d_values else np.nan,
        })

    pair_df = pd.DataFrame(records)
    if pair_df.empty:
        return pair_df, pd.DataFrame()

    pair_df["distance_bin_start"] = (pair_df["distance_bp"] // distance_bin_bp) * distance_bin_bp
    pair_df["distance_bin_end"] = pair_df["distance_bin_start"] + distance_bin_bp
    binned = pair_df.groupby(["distance_bin_start", "distance_bin_end"], as_index=False).agg(
        n_pairs=("r2", "size"),
        mean_r2=("r2", "mean"),
        median_r2=("r2", "median"),
        q25_r2=("r2", lambda x: np.nanquantile(x, 0.25)),
        q75_r2=("r2", lambda x: np.nanquantile(x, 0.75)),
    )
    binned["distance_mid_bp"] = 0.5 * (binned["distance_bin_start"] + binned["distance_bin_end"])
    summary = pd.DataFrame(per_chr_summary)
    return pair_df, binned.merge(summary.assign(key=1), how="cross") if False else binned



def estimate_ld_window(ld_bins: pd.DataFrame, threshold_r2: float = 0.2) -> pd.DataFrame:
    if ld_bins.empty:
        return pd.DataFrame([{
            "threshold_r2": threshold_r2,
            "ld_decay_bp": np.nan,
            "half_max_decay_bp": np.nan,
            "max_observed_mean_r2": np.nan,
        }])
    bins = ld_bins.sort_values("distance_mid_bp").copy()
    max_r2 = float(bins["mean_r2"].max())
    below = bins[bins["mean_r2"] <= threshold_r2]
    below_half = bins[bins["mean_r2"] <= max_r2 / 2.0]
    return pd.DataFrame([{
        "threshold_r2": threshold_r2,
        "ld_decay_bp": int(below.iloc[0]["distance_mid_bp"]) if not below.empty else np.nan,
        "half_max_decay_bp": int(below_half.iloc[0]["distance_mid_bp"]) if not below_half.empty else np.nan,
        "max_observed_mean_r2": max_r2,
    }])



def ld_prune_greedy(
    geno_df: pd.DataFrame,
    snp_info: pd.DataFrame,
    window_bp: int = 250_000,
    r2_threshold: float = 0.8,
    max_snps_per_chr: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    shared = [s for s in snp_info.index if s in geno_df.columns]
    geno = geno_df[shared].copy()
    info = snp_info.loc[shared].copy()
    _, _, Z = mean_impute_and_standardize(geno)
    n = Z.shape[0]

    keep_all: List[str] = []
    rows = []
    col_index = {snp: i for i, snp in enumerate(shared)}
    for chrom in sorted(info["chrom"].dropna().astype(str).unique(), key=lambda x: int(re.search(r"\d+", x).group(0)) if re.search(r"\d+", x) else x):
        sub = info[info["chrom"].astype(str) == str(chrom)].sort_values("pos")
        if max_snps_per_chr and sub.shape[0] > max_snps_per_chr:
            step = max(1, int(math.ceil(sub.shape[0] / max_snps_per_chr)))
            sub = sub.iloc[::step, :].copy()
        keep_idx: List[int] = []
        keep_pos: List[int] = []
        for snp, pos in zip(sub.index, sub["pos"].astype(int)):
            j = col_index[snp]
            candidate = True
            if keep_idx:
                in_window = [k for k, p in zip(keep_idx, keep_pos) if abs(pos - p) <= window_bp]
                if in_window:
                    corrs = np.dot(Z[:, in_window].T, Z[:, j]) / max(n - 1, 1)
                    if np.any(np.square(corrs) > r2_threshold):
                        candidate = False
            if candidate:
                keep_idx.append(j)
                keep_pos.append(pos)
                keep_all.append(snp)
        rows.append({
            "chrom": str(chrom),
            "n_input_snps": int(sub.shape[0]),
            "n_kept_snps": int(len(keep_idx)),
            "fraction_kept": float(len(keep_idx) / max(sub.shape[0], 1)),
            "window_bp": int(window_bp),
            "r2_threshold": float(r2_threshold),
        })
    return pd.DataFrame(rows), keep_all



def compute_local_ld_matrix(
    geno_df: pd.DataFrame,
    snp_order: Sequence[str],
    max_snps: int = 80,
) -> pd.DataFrame:
    snps = [s for s in snp_order if s in geno_df.columns]
    if not snps:
        return pd.DataFrame()
    if len(snps) > max_snps:
        idx = np.linspace(0, len(snps) - 1, max_snps, dtype=int)
        snps = [snps[i] for i in idx]
    _, _, Z = mean_impute_and_standardize(geno_df[snps])
    corr = np.dot(Z.T, Z) / max(Z.shape[0] - 1, 1)
    r2 = np.square(corr)
    out = pd.DataFrame(r2, index=snps, columns=snps)
    return out


# ---------------------------------------------------------------------------
# PCA, correlation, diagnostics
# ---------------------------------------------------------------------------


def compute_genotype_pca(geno_df: pd.DataFrame, n_components: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exact PCA using the sample-space covariance matrix.

    This is much faster than fitting a full feature-space PCA when the number of
    SNPs greatly exceeds the number of accessions (the common case here).
    """
    _, _, Z = mean_impute_and_standardize(geno_df)
    n_samples, n_features = Z.shape
    n_components = min(n_components, n_samples, n_features)
    if n_components <= 0:
        return pd.DataFrame(index=geno_df.index), pd.DataFrame()

    # Sample-space covariance / Gram matrix: n_samples x n_samples
    gram = np.dot(Z, Z.T) / max(n_features - 1, 1)
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    evecs = evecs[:, order]
    evals = evals[:n_components]
    evecs = evecs[:, :n_components]

    # Scores correspond to U * S where covariance eigenvalues = S^2 / (p - 1)
    singular_values = np.sqrt(evals * max(n_features - 1, 1))
    scores = evecs * singular_values

    total_var = float(np.maximum(evals.sum(), 0.0))
    explained_ratio = evals / total_var if total_var > 0 else np.zeros_like(evals)

    score_df = pd.DataFrame(scores, index=geno_df.index, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
    scree = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(evals))],
        "component_index": np.arange(1, len(evals) + 1),
        "explained_variance": evals,
        "explained_variance_ratio": explained_ratio,
        "cumulative_variance_ratio": np.cumsum(explained_ratio),
    })
    return score_df, scree



def compute_correlation_matrices(
    phenotype_df: pd.DataFrame,
    trait_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [c for c in trait_columns if c in phenotype_df.columns]
    if len(cols) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    r_mat = pd.DataFrame(np.nan, index=cols, columns=cols)
    p_mat = pd.DataFrame(np.nan, index=cols, columns=cols)
    n_mat = pd.DataFrame(0, index=cols, columns=cols)
    long_rows = []
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            x = pd.to_numeric(phenotype_df[c1], errors="coerce")
            y = pd.to_numeric(phenotype_df[c2], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            if n < 3:
                continue
            if i == j:
                r, p = 1.0, np.nan
            else:
                r, p = stats.pearsonr(x[mask], y[mask])
            r_mat.loc[c1, c2] = r
            p_mat.loc[c1, c2] = p
            n_mat.loc[c1, c2] = n
            star = ""
            if i != j and np.isfinite(p):
                if p < 0.001:
                    star = "***"
                elif p < 0.01:
                    star = "**"
                elif p < 0.05:
                    star = "*"
            long_rows.append({
                "trait_1": c1,
                "trait_2": c2,
                "r": r,
                "p": p,
                "n": n,
                "significance": star,
                "test": "Pearson correlation (two-sided)",
                "diagonal_test_performed": bool(i != j),
            })
    long_df = pd.DataFrame(long_rows)
    return r_mat, p_mat, long_df



def gwas_diagnostic_summary(assoc_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for trait, df in assoc_tables.items():
        if df.empty:
            continue
        p = df["p"].astype(float).clip(lower=np.nextafter(0, 1), upper=1)
        m = p.shape[0]
        bonf = 0.05 / max(m, 1)
        chi2_vals = stats.chi2.isf(p, df=1)
        lambda_gc = float(np.nanmedian(chi2_vals) / stats.chi2.ppf(0.5, df=1)) if len(chi2_vals) else np.nan
        rows.append({
            "trait": trait,
            "n_snps": int(m),
            "bonferroni_p": bonf,
            "n_bonferroni": int((p <= bonf).sum()),
            "fraction_bonferroni": float((p <= bonf).mean()),
            "n_p_lt_1e-5": int((p <= 1e-5).sum()),
            "fraction_p_lt_1e-5": float((p <= 1e-5).mean()),
            "median_minus_log10p": float(np.median(-np.log10(p))),
            "max_minus_log10p": float(np.max(-np.log10(p))),
            "lambda_gc": lambda_gc,
        })
    return pd.DataFrame(rows).sort_values(["n_bonferroni", "max_minus_log10p"], ascending=[False, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# QTL calling and region summaries
# ---------------------------------------------------------------------------


def call_qtls(
    assoc_df: pd.DataFrame,
    window_bp: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    if assoc_df.empty:
        return pd.DataFrame()
    m = assoc_df.shape[0]
    bonf = alpha / max(m, 1)
    sig = assoc_df[assoc_df["p"] <= bonf].copy().sort_values(["chrom", "pos", "p"])
    if sig.empty:
        return pd.DataFrame(columns=["trait", "chrom", "qtl_start", "qtl_end", "lead_snp", "lead_pos", "lead_p", "n_sig_snps", "bonferroni_p", "window_bp"])

    rows = []
    locus_counter = 0
    for chrom in sorted(sig["chrom"].astype(str).unique(), key=lambda x: int(re.search(r"\d+", x).group(0)) if re.search(r"\d+", x) else x):
        sub = sig[sig["chrom"].astype(str) == str(chrom)].sort_values("pos").reset_index(drop=True)
        i = 0
        while i < sub.shape[0]:
            start_anchor = int(sub.loc[i, "pos"])
            j = i + 1
            while j < sub.shape[0] and (int(sub.loc[j, "pos"]) - start_anchor) <= window_bp:
                j += 1
            block = sub.iloc[i:j].copy()
            lead = block.loc[block["p"].idxmin()]
            locus_counter += 1
            rows.append({
                "trait": str(lead["trait"]),
                "locus_id": f"{chrom}_{locus_counter}",
                "chrom": str(chrom),
                "qtl_start": int(max(0, int(lead["pos"]) - window_bp)),
                "qtl_end": int(int(lead["pos"]) + window_bp),
                "lead_snp": str(lead["snp"]),
                "lead_pos": int(lead["pos"]),
                "lead_p": float(lead["p"]),
                "lead_minus_log10p": float(lead["minus_log10p"]),
                "n_sig_snps": int(block.shape[0]),
                "bonferroni_p": float(bonf),
                "window_bp": int(window_bp),
            })
            i = j
    return pd.DataFrame(rows)



def annotate_qtls_with_genes(
    qtl_df: pd.DataFrame,
    gene_df: pd.DataFrame,
    highlight_terms: Sequence[str] = DEFAULT_HIGHLIGHT_GENES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if qtl_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    gene_df = gene_df.copy()
    gene_df["chrom"] = gene_df["chrom"].map(normalize_chr)
    long_rows = []
    summary_rows = []
    for _, qtl in qtl_df.iterrows():
        chrom = str(qtl["chrom"])
        start = int(qtl["qtl_start"])
        end = int(qtl["qtl_end"])
        lead = int(qtl["lead_pos"])
        genes = gene_df[(gene_df["chrom"].astype(str) == chrom) & (gene_df["start"] <= end) & (gene_df["end"] >= start)].copy()
        if genes.empty:
            nearest = gene_df[gene_df["chrom"].astype(str) == chrom].copy()
            if nearest.empty:
                summary_rows.append({**qtl.to_dict(), "nearest_gene": "", "genes_in_window": "", "n_genes_in_window": 0, "highlight_genes_detected": ""})
                continue
            dist = np.minimum(np.abs(nearest["start"] - lead), np.abs(nearest["end"] - lead))
            nearest = nearest.assign(dist_to_lead=dist).sort_values("dist_to_lead")
            ng = nearest.iloc[0]
            summary_rows.append({**qtl.to_dict(), "nearest_gene": ng["gene_id"], "genes_in_window": "", "n_genes_in_window": 0, "highlight_genes_detected": ""})
            continue
        genes = genes.copy()
        genes["dist_to_lead"] = np.minimum(np.abs(genes["start"] - lead), np.abs(genes["end"] - lead))
        def matches_highlight(row) -> bool:
            haystack = " | ".join([str(row.get("gene_id", "")), str(row.get("gene_name", "")), str(row.get("description", "")), str(row.get("aliases", ""))]).lower()
            return any(term.lower() in haystack for term in highlight_terms)
        genes["is_highlight_gene"] = genes.apply(matches_highlight, axis=1)
        genes = genes.sort_values(["dist_to_lead", "start"]).reset_index(drop=True)
        for _, g in genes.iterrows():
            long_rows.append({
                **qtl.to_dict(),
                "gene_id": g["gene_id"],
                "gene_name": g.get("gene_name", g["gene_id"]),
                "gene_start": int(g["start"]),
                "gene_end": int(g["end"]),
                "dist_to_lead": int(g["dist_to_lead"]),
                "description": g.get("description", ""),
                "aliases": g.get("aliases", ""),
                "is_highlight_gene": bool(g["is_highlight_gene"]),
            })
        highlights = genes.loc[genes["is_highlight_gene"], "gene_id"].astype(str).tolist()
        summary_rows.append({
            **qtl.to_dict(),
            "nearest_gene": str(genes.iloc[0]["gene_id"]),
            "genes_in_window": "; ".join(genes["gene_id"].astype(str).tolist()),
            "n_genes_in_window": int(genes.shape[0]),
            "highlight_genes_detected": "; ".join(highlights),
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(long_rows)



def select_focus_locus(
    assoc_tables: Dict[str, pd.DataFrame],
    focus_chrom: str = "6",
    preferred_keywords: Sequence[str] = ("entropy", "748", "750", "nir"),
) -> Tuple[Optional[str], Optional[pd.Series]]:
    candidates = []
    for trait, df in assoc_tables.items():
        if df.empty:
            continue
        sub = df[df["chrom"].astype(str) == str(focus_chrom)].copy()
        if sub.empty:
            continue
        lead = sub.loc[sub["p"].idxmin()].copy()
        score = 0
        t = trait.lower()
        for i, kw in enumerate(preferred_keywords):
            if kw.lower() in t:
                score += 100 - i
        candidates.append((score, float(lead["p"]), trait, lead))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    _, _, trait, lead = candidates[0]
    return trait, lead



def build_local_region_tables(
    assoc_df: pd.DataFrame,
    geno_df: pd.DataFrame,
    snp_info: pd.DataFrame,
    gene_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    focus_trait: str,
    focus_lead_snp: Optional[str] = None,
    focus_chrom: Optional[str] = None,
    focus_pos: Optional[int] = None,
    window_bp: int = 250_000,
    ld_max_snps: int = 80,
    haplotype_max_snps: int = 4,
    haplotype_r2_threshold: float = 0.6,
    highlight_terms: Sequence[str] = DEFAULT_HIGHLIGHT_GENES,
) -> Dict[str, pd.DataFrame]:
    if assoc_df.empty:
        return {"local_assoc": pd.DataFrame(), "local_ld": pd.DataFrame(), "local_genes": pd.DataFrame(), "haplotypes": pd.DataFrame(), "haplotype_trait_summary": pd.DataFrame(), "local_summary": pd.DataFrame()}

    work = assoc_df.copy()
    if focus_lead_snp is not None and focus_lead_snp in set(work["snp"]):
        lead = work.loc[work["snp"] == focus_lead_snp].sort_values("p").iloc[0]
    elif focus_chrom is not None and focus_pos is not None:
        sub = work[work["chrom"].astype(str) == str(focus_chrom)].copy()
        if sub.empty:
            raise RuntimeError(f"No association rows found on chromosome {focus_chrom}")
        lead = sub.iloc[(sub["pos"] - focus_pos).abs().argmin()]
    else:
        lead = work.loc[work["p"].idxmin()]
    chrom = str(lead["chrom"])
    pos = int(lead["pos"])
    region_start = max(0, pos - window_bp)
    region_end = pos + window_bp

    local_assoc = work[(work["chrom"].astype(str) == chrom) & (work["pos"].between(region_start, region_end))].copy().sort_values("pos")

    local_snp_info = snp_info[(snp_info["chrom"].astype(str) == chrom) & (pd.to_numeric(snp_info["pos"], errors="coerce").between(region_start, region_end))].copy().sort_values("pos")
    local_snps = [s for s in local_snp_info.index if s in geno_df.columns]
    local_ld = compute_local_ld_matrix(geno_df[local_snps], local_snps, max_snps=ld_max_snps) if local_snps else pd.DataFrame()

    gene_region = gene_df[(gene_df["chrom"].astype(str) == chrom) & (gene_df["end"] >= region_start) & (gene_df["start"] <= region_end)].copy().sort_values("start")
    if not gene_region.empty:
        def highlight_gene(row) -> bool:
            haystack = " | ".join([str(row.get("gene_id", "")), str(row.get("gene_name", "")), str(row.get("description", "")), str(row.get("aliases", ""))]).lower()
            return any(term.lower() in haystack for term in highlight_terms)
        gene_region["is_highlight_gene"] = gene_region.apply(highlight_gene, axis=1)
        gene_region["dist_to_lead"] = np.minimum(np.abs(gene_region["start"] - pos), np.abs(gene_region["end"] - pos))
    else:
        gene_region = pd.DataFrame(columns=["gene_id", "chrom", "start", "end", "gene_name", "description", "aliases", "is_highlight_gene", "dist_to_lead"])

    # Haplotype grouping using lead SNP + strongest LD partners in the region.
    hap_snps = []
    if local_snps:
        lead_snp = str(lead["snp"])
        if lead_snp not in local_snps:
            # choose nearest genotyped SNP
            local_snp_info2 = local_snp_info.copy()
            local_snp_info2["abs_dist"] = (local_snp_info2["pos"] - pos).abs()
            lead_snp = str(local_snp_info2.sort_values("abs_dist").index[0])
        hap_snps = [lead_snp]
        lead_g = geno_df[lead_snp].to_numpy(dtype=np.float32)
        ranked = []
        for snp in local_snps:
            if snp == lead_snp:
                continue
            r2 = pairwise_r2(lead_g, geno_df[snp].to_numpy(dtype=np.float32))
            if np.isfinite(r2):
                ranked.append((r2, snp))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        strong = [s for r2, s in ranked if r2 >= haplotype_r2_threshold][: max(0, haplotype_max_snps - 1)]
        if len(strong) < max(0, haplotype_max_snps - 1):
            strong += [s for _, s in ranked if s not in strong][: max(0, haplotype_max_snps - 1 - len(strong))]
        hap_snps.extend(strong)
        hap_snps = hap_snps[:haplotype_max_snps]

    haplotypes = pd.DataFrame()
    hap_summary = pd.DataFrame()
    if hap_snps:
        g = geno_df[hap_snps].copy()
        g = g.apply(pd.to_numeric, errors="coerce")
        enc = g.apply(lambda col: col.map(lambda v: "NA" if pd.isna(v) else str(int(round(float(v))))))
        hap = enc.agg("-".join, axis=1)
        haplotypes = pd.DataFrame({
            "accession": geno_df.index,
            "haplotype": hap.values,
        })
        for snp in hap_snps:
            haplotypes[snp] = g[snp].values
        trait_col = find_trait_column(phenotype_df, focus_trait)
        if trait_col is None:
            trait_col = find_trait_column(phenotype_df, lead["trait"])
        if trait_col is not None:
            ph = pd.to_numeric(phenotype_df.loc[haplotypes["accession"], trait_col], errors="coerce")
            haplotypes["phenotype_col"] = trait_col
            haplotypes["phenotype_value"] = ph.values
            summary = haplotypes.groupby("haplotype", as_index=False).agg(
                n_accessions=("accession", "size"),
                phenotype_mean=("phenotype_value", "mean"),
                phenotype_sd=("phenotype_value", "std"),
                phenotype_median=("phenotype_value", "median"),
            )
            summary["frequency"] = summary["n_accessions"] / max(summary["n_accessions"].sum(), 1)
            hap_summary = summary.sort_values(["n_accessions", "haplotype"], ascending=[False, True]).reset_index(drop=True)

    local_summary = pd.DataFrame([{
        "focus_trait": focus_trait,
        "lead_trait_row": str(lead["trait"]),
        "lead_snp": str(lead["snp"]),
        "chrom": chrom,
        "lead_pos": pos,
        "lead_p": float(lead["p"]),
        "window_bp": int(window_bp),
        "region_start": int(region_start),
        "region_end": int(region_end),
        "n_assoc_snps_in_region": int(local_assoc.shape[0]),
        "n_genotyped_snps_in_region": int(len(local_snps)),
        "n_genes_in_region": int(gene_region.shape[0]),
        "haplotype_snps": "; ".join(hap_snps),
    }])

    return {
        "local_assoc": local_assoc,
        "local_ld": local_ld,
        "local_genes": gene_region,
        "haplotypes": haplotypes,
        "haplotype_trait_summary": hap_summary,
        "local_summary": local_summary,
    }


# ---------------------------------------------------------------------------
# Supplement/inter-seed helpers
# ---------------------------------------------------------------------------


def load_interseed_overview(assets: AssetDiscovery) -> pd.DataFrame:
    if not (assets.supplement_xlsx and assets.supplement_xlsx.exists()):
        return pd.DataFrame()
    xls = pd.ExcelFile(assets.supplement_xlsx)
    if "CandidateGenes_interSeed" not in xls.sheet_names:
        return pd.DataFrame()
    df = pd.read_excel(xls, sheet_name="CandidateGenes_interSeed")
    col_map = {c.lower(): c for c in df.columns}
    trait_col = col_map.get("trait")
    gene_col = col_map.get("gene_id") or col_map.get("gene id")
    desc_col = col_map.get("description")
    dist_col = col_map.get("distance_from_lead")
    if not all([trait_col, gene_col]):
        return df
    out = df.copy()
    out[trait_col] = out[trait_col].astype(str)
    out[gene_col] = out[gene_col].astype(str)
    if dist_col:
        out[dist_col] = pd.to_numeric(out[dist_col], errors="coerce")
    # concise ranked view for manuscript support
    grouped = []
    for trait, sub in out.groupby(trait_col):
        if dist_col:
            sub = sub.sort_values(dist_col)
        seen = set()
        keep = []
        for _, row in sub.iterrows():
            gid = row[gene_col]
            if gid in seen:
                continue
            seen.add(gid)
            keep.append(row)
            if len(keep) >= 25:
                break
        for row in keep:
            grouped.append({
                "Trait": trait,
                "Gene_ID": row[gene_col],
                "Distance_from_Lead": row[dist_col] if dist_col else np.nan,
                "Description": row[desc_col] if desc_col else "",
            })
    return pd.DataFrame(grouped)



def summarize_interseed_heterogeneity(phenotype_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in phenotype_df.columns if any(c.endswith(suf) for suf in ["_CV", "_Std", "_Skew", "_p10", "_p90"])]
    if not cols:
        return pd.DataFrame()
    rows = []
    for col in cols:
        x = pd.to_numeric(phenotype_df[col], errors="coerce")
        rows.append({
            "metric": col,
            "n": int(np.isfinite(x).sum()),
            "mean": float(np.nanmean(x)),
            "sd": float(np.nanstd(x, ddof=1)),
            "median": float(np.nanmedian(x)),
            "q10": float(np.nanquantile(x, 0.10)),
            "q90": float(np.nanquantile(x, 0.90)),
        })
    return pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def autosize_openpyxl_worksheet(ws) -> None:
    for col_cells in ws.columns:
        length = 0
        col_letter = col_cells[0].column_letter
        for cell in col_cells:
            val = "" if cell.value is None else str(cell.value)
            length = max(length, min(60, len(val)))
        ws.column_dimensions[col_letter].width = max(10, length + 2)



def write_workbook(path: Path, sheet_dict: Dict[str, pd.DataFrame]) -> None:
    """Write a reviewer-facing Excel workbook quickly using xlsxwriter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        workbook = writer.book
        header_fmt = workbook.add_format({
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#1F4E78",
            "align": "center",
            "valign": "vcenter",
            "text_wrap": False,
        })
        for sheet_name, df in sheet_dict.items():
            safe_name = slugify(sheet_name)[:31] or "Sheet1"
            out = df.copy()
            write_index = not isinstance(out.index, pd.RangeIndex)
            out.to_excel(writer, sheet_name=safe_name, index=write_index)
            ws = writer.sheets[safe_name]
            ws.freeze_panes(1, 0)
            ws.hide_gridlines(2)
            nrows, ncols = out.shape
            extra_cols = 1 if write_index else 0
            # Rewrite headers with formatting
            headers = []
            if write_index:
                headers.append(out.index.name or "index")
            headers.extend([str(c) for c in out.columns])
            for col_idx, header in enumerate(headers):
                ws.write(0, col_idx, header, header_fmt)
                try:
                    series = out.iloc[:, col_idx - extra_cols] if col_idx >= extra_cols else pd.Series(out.index.astype(str))
                    max_len = max([len(str(header))] + [len(str(x)) for x in series.head(1000)])
                except Exception:
                    max_len = len(str(header))
                ws.set_column(col_idx, col_idx, min(max(10, max_len + 2), 40))
            if nrows >= 1 and len(headers) >= 1:
                ws.autofilter(0, 0, nrows, len(headers) - 1)



def write_csv_bundle(output_dir: Path, sheet_dict: Dict[str, pd.DataFrame]) -> None:
    ensure_dir(output_dir)
    for sheet_name, df in sheet_dict.items():
        csv_path = output_dir / f"{slugify(sheet_name)}.csv"
        if isinstance(df.index, pd.RangeIndex):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path)



def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)



def analysis_coverage_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"analysis_topic": "LD window / LD pruning", "analysis_output": "ld_decay_bins, ld_decay_summary, ld_pruning_summary", "notes": "Estimates empirical LD decay and sample-aware pruning summary."},
        {"analysis_topic": "chr6 locus relative to 250 kb window and nearby genes", "analysis_output": "chr6_local_summary, chr6_local_genes, chr6_local_assoc", "notes": "Directly shows lead SNP, region bounds, and overlapping genes."},
        {"analysis_topic": "local LD heatmap / haplotype support", "analysis_output": "chr6_local_ld, chr6_haplotypes, chr6_haplotype_trait_summary", "notes": "Supports local linkage interpretation around the lead locus."},
        {"analysis_topic": "main-text significant peak table", "analysis_output": "qtl_peak_summary, qtl_gene_long", "notes": "Trait, peak, window, lead SNP, nearest and overlapping genes."},
        {"analysis_topic": "justification for 10 genotype PCs", "analysis_output": "geno_pca_scree", "notes": "Explained variance and cumulative variance for scree plot."},
        {"analysis_topic": "heatmap significance / no diagonal test", "analysis_output": "correlation_r, correlation_p, correlation_long", "notes": "Diagonal p-values are intentionally omitted."},
        {"analysis_topic": "why many SNPs are significant for some traits", "analysis_output": "gwas_diagnostics", "notes": "Signal burden and lambda_GC across traits."},
        {"analysis_topic": "spectrum heterogeneity / inter-seed interpretation", "analysis_output": "heterogeneity_summary, interseed_overview", "notes": "Summarizes spread/skew metrics and inter-seed candidate-gene overview."},
    ])
