'''
The following code is from a public notebook on Kaggle for Stanford RNA 3D Folding Part 2.
https://www.kaggle.com/code/alisalmanrana/fork-of-fork-of-template-or-protenix-lb-0-43-ish
'''

# ── stdlib / env ──────────────────────────────────────────────────────────────
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & PATHS
# ══════════════════════════════════════════════════════════════════════════════
IS_KAGGLE = True

DATA_BASE         = "/kaggle/input/stanford-rna-3d-folding-2"
DEFAULT_TEST_CSV  = f"{DATA_BASE}/test_sequences.csv"
DEFAULT_TRAIN_CSV = f"{DATA_BASE}/train_sequences.csv"
DEFAULT_TRAIN_LBLS= f"{DATA_BASE}/train_labels.csv"
DEFAULT_VAL_CSV   = f"{DATA_BASE}/validation_sequences.csv"
DEFAULT_VAL_LBLS  = f"{DATA_BASE}/validation_labels.csv"
DEFAULT_OUTPUT    = "/kaggle/working/submission.csv"

DEFAULT_CODE_DIR = (
    "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted"
    "/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"
)
DEFAULT_ROOT_DIR = DEFAULT_CODE_DIR

MODEL_NAME    = "protenix_base_20250630_v1.0.0"
N_SAMPLE      = 5
SEED          = 42
MAX_SEQ_LEN   = int(os.environ.get("MAX_SEQ_LEN",   "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP",  "192"))  # v17: 128→192 for smoother blending

# ★ Lowered from 50 → 45 to catch more partial-homology templates
MIN_SIMILARITY       = float(os.environ.get("MIN_SIMILARITY",       "0.0"))
# v18: restored to 50% (reference threshold) — 45% with top_n=50 filled all slots with
# garbage templates, preventing Protenix from running. 50% naturally routes hard targets
# (9JFS, 9LEC, 9LEL) to Protenix when no high-quality template exists.
MIN_PERCENT_IDENTITY = float(os.environ.get("MIN_PERCENT_IDENTITY", "50.0"))

USE_PROTENIX = True


def parse_bool(v, default=False):
    s = str(v).strip().lower()
    if s in {"1","true","t","yes","y","on"}:  return "true"
    if s in {"0","false","f","no","n","off"}: return "false"
    return "true" if default else "false"


USE_MSA      = parse_bool(os.environ.get("USE_MSA",      "false"))
USE_TEMPLATE = parse_bool(os.environ.get("USE_TEMPLATE", "false"))
USE_RNA_MSA  = parse_bool(os.environ.get("USE_RNA_MSA",  "true"))
MODEL_N_SAMPLE = int(os.environ.get("MODEL_N_SAMPLE", str(N_SAMPLE)))


# ══════════════════════════════════════════════════════════════════════════════
# GENERAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def seed_everything(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark    = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled      = True
    try:    torch.use_deterministic_algorithms(True)
    except: pass


def resolve_paths():
    return (
        os.environ.get("TEST_CSV",          DEFAULT_TEST_CSV),
        os.environ.get("SUBMISSION_CSV",    DEFAULT_OUTPUT),
        os.environ.get("PROTENIX_CODE_DIR", DEFAULT_CODE_DIR),
        os.environ.get("PROTENIX_ROOT_DIR", DEFAULT_ROOT_DIR),
    )


def ensure_required_files(root_dir):
    for p, name in [
        (Path(root_dir)/"checkpoint"/f"{MODEL_NAME}.pt",         "checkpoint"),
        (Path(root_dir)/"common"/"components.cif",               "CCD file"),
        (Path(root_dir)/"common"/"components.cif.rdkit_mol.pkl", "CCD cache"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {name}: {p}")


def build_input_json(df, json_path):
    data = [
        {"name": row["target_id"], "covalent_bonds": [],
         "sequences": [{"rnaSequence": {"sequence": row["sequence"], "count": 1}}]}
        for _, row in df.iterrows()
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def build_configs(input_json_path, dump_dir, model_name, n_sample=None, seed=None):
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs

    ns   = n_sample if n_sample is not None else MODEL_N_SAMPLE
    use_seed = seed if seed is not None else SEED
    base = {**configs_base, **{"data": data_configs}, **inference_configs}

    def deep_update(t, p):
        for k, v in p.items():
            if isinstance(v, dict) and k in t and isinstance(t[k], dict):
                deep_update(t[k], v)
            else:
                t[k] = v

    deep_update(base, model_configs[model_name])
    arg_str = " ".join([
        f"--model_name {model_name}",
        f"--input_json_path {input_json_path}",
        f"--dump_dir {dump_dir}",
        f"--use_msa {USE_MSA}",
        f"--use_template {USE_TEMPLATE}",
        f"--use_rna_msa {USE_RNA_MSA}",
        f"--sample_diffusion.N_sample {ns}",
        f"--seeds {use_seed}",
    ])
    return parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)


def coords_to_rows(target_id, seq, coords):
    """coords: (N_SAMPLE, seq_len, 3)"""
    rows = []
    for i in range(len(seq)):
        row = {"ID": f"{target_id}_{i+1}", "resname": seq[i], "resid": i+1}
        for s in range(N_SAMPLE):
            if s < coords.shape[0] and i < coords.shape[1]:
                x, y, z = coords[s, i]
            else:
                x, y, z = 0.0, 0.0, 0.0
            row[f"x_{s+1}"] = float(x)
            row[f"y_{s+1}"] = float(y)
            row[f"z_{s+1}"] = float(z)
        rows.append(row)
    return rows


# ── chunking / stitching ─────────────────────────────────────────────────────
def split_into_chunks(seq_len, max_len, overlap):
    if seq_len <= max_len:
        return [(0, seq_len)]
    chunks, step, pos = [], max_len - overlap, 0
    while pos < seq_len:
        end = min(pos + max_len, seq_len)
        chunks.append((pos, end))
        if end == seq_len: break
        pos += step
    return chunks


def kabsch_align(P, Q):
    cP, cQ = P.mean(0), Q.mean(0)
    Pc, Qc = P - cP, Q - cQ
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.eye(3)
    if d < 0: S[2,2] = -1
    R = Vt.T @ S @ U.T
    return R, cQ - R @ cP


def stitch_chunk_coords(chunk_coords_list, chunk_ranges, seq_len):
    if len(chunk_coords_list) == 1:
        c = chunk_coords_list[0]
        out = np.zeros((seq_len, 3), dtype=c.dtype)
        out[:min(c.shape[0], seq_len)] = c[:min(c.shape[0], seq_len)]
        return out

    aligned = [chunk_coords_list[0].copy()]
    for i in range(1, len(chunk_coords_list)):
        ps, pe = chunk_ranges[i-1]
        cs, ce = chunk_ranges[i]
        ov_s, ov_e = cs, min(pe, ce)
        if ov_e - ov_s < 3:
            aligned.append(chunk_coords_list[i].copy()); continue
        prev_ov = aligned[i-1][ov_s-ps:ov_e-ps]
        cur_ov  = chunk_coords_list[i][ov_s-cs:ov_e-cs]
        valid   = ~(np.isnan(prev_ov).any(1) | np.isnan(cur_ov).any(1))
        if valid.sum() < 3:
            aligned.append(chunk_coords_list[i].copy()); continue
        R, t = kabsch_align(cur_ov[valid], prev_ov[valid])
        aligned.append((chunk_coords_list[i] @ R.T) + t)

    full    = np.zeros((seq_len, 3), dtype=np.float64)
    weights = np.zeros(seq_len,     dtype=np.float64)
    for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
        cl  = coords.shape[0]
        ae  = min(s + cl, seq_len)
        ul  = ae - s
        w   = np.ones(ul, dtype=np.float64)
        if i > 0:
            ov_e2 = min(chunk_ranges[i-1][1], e); rl = ov_e2 - s
            if rl > 0: w[:rl] = np.linspace(0., 1., rl)
        if i < len(chunk_ranges) - 1:
            ns2 = chunk_ranges[i+1][0]; rs = ns2 - s; rl = ae - ns2
            if rl > 0 and rs < ul: w[rs:ul] = np.linspace(1., 0., rl)
        full[s:ae]    += coords[:ul] * w[:, None]
        weights[s:ae] += w
    mask = weights > 0
    full[mask] /= weights[mask, None]
    return full


# ══════════════════════════════════════════════════════════════════════════════
# TBM CORE  — with ID-suffix sort fix + exact-match index
# ══════════════════════════════════════════════════════════════════════════════
def _make_aligner():
    al = PairwiseAligner()
    al.mode             = "global"
    al.match_score      = 2
    al.mismatch_score   = -1.5
    al.open_gap_score   = -8
    al.extend_gap_score = -0.4
    for attr in [
        "query_left_open_gap_score",  "query_left_extend_gap_score",
        "query_right_open_gap_score", "query_right_extend_gap_score",
        "target_left_open_gap_score", "target_left_extend_gap_score",
        "target_right_open_gap_score","target_right_extend_gap_score",
    ]:
        try: setattr(al, attr, -8 if "open" in attr else -0.4)
        except: pass
    return al

_aligner = _make_aligner()


def parse_stoichiometry(stoich):
    if pd.isna(stoich) or str(stoich).strip() == "": return []
    return [(ch.strip(), int(cnt))
            for part in str(stoich).split(";")
            for ch, cnt in [part.split(":")]]


def parse_fasta(fasta_content):
    out, cur, parts = {}, None, []
    for line in str(fasta_content).splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            if cur is not None: out[cur] = "".join(parts)
            cur, parts = line[1:].split()[0], []
        else: parts.append(line.replace(" ",""))
    if cur is not None: out[cur] = "".join(parts)
    return out


def get_chain_segments(row):
    seq    = row["sequence"]
    stoich = row.get("stoichiometry", "")
    all_sq = row.get("all_sequences", "")
    if pd.isna(stoich) or pd.isna(all_sq) or str(stoich).strip()=="" or str(all_sq).strip()=="":
        return [(0, len(seq))]
    try:
        cd    = parse_fasta(all_sq)
        order = parse_stoichiometry(stoich)
        segs, pos = [], 0
        for ch, cnt in order:
            base = cd.get(ch)
            if base is None: return [(0, len(seq))]
            for _ in range(cnt):
                segs.append((pos, pos+len(base))); pos += len(base)
        return segs if pos == len(seq) else [(0, len(seq))]
    except: return [(0, len(seq))]


def build_segments_map(df):
    seg_map, stoich_map = {}, {}
    for _, r in df.iterrows():
        tid           = r["target_id"]
        seg_map[tid]  = get_chain_segments(r)
        raw_s         = r.get("stoichiometry","")
        stoich_map[tid] = "" if pd.isna(raw_s) else str(raw_s)
    return seg_map, stoich_map


def process_labels(labels_df):
    """
    ★★★ BUG-FIX: Sort by ID SUFFIX INTEGER, not resid column.

    For multi-copy targets (9MME=8×580nt, 9JGM, 9G4P, 9G4Q, 9I9W, 9WHV, 9ZCC),
    resid RESETS to 1 at the start of each copy.
    sort_values("resid") SCRAMBLES coordinates → TM≈0.1.
    ID suffix "9MME_581" → 581 is always sequential → correct order → TM≈0.9+.
    """
    coords = {}
    df = labels_df.copy()
    split      = df["ID"].str.rsplit("_", n=1)
    df["_pfx"] = split.str[0]
    df["_pos"] = pd.to_numeric(split.str[1], errors="coerce").fillna(0).astype(int)
    for prefix, grp in df.groupby("_pfx"):
        arr = grp.sort_values("_pos")[["x_1","y_1","z_1"]].values.astype(np.float64)
        arr[np.abs(arr) > 1e10] = np.nan
        for i in range(len(arr)):
            if not np.isnan(arr[i,0]): continue
            pv = next((j for j in range(i-1,-1,-1) if not np.isnan(arr[j,0])), -1)
            nv = next((j for j in range(i+1,len(arr)) if not np.isnan(arr[j,0])), -1)
            if   pv>=0 and nv>=0: w=( i-pv)/(nv-pv); arr[i]=(1-w)*arr[pv]+w*arr[nv]
            elif pv>=0:           arr[i] = arr[pv]+[3,0,0]
            elif nv>=0:           arr[i] = arr[nv]+[3,0,0]
            else:                 arr[i] = [i*3,0,0]
        coords[prefix] = np.nan_to_num(arr, nan=0.0)
    return coords


def _coords_are_valid(c):
    """Filter collapsed / zero / degenerate template coordinates."""
    if c is None or not np.all(np.isfinite(c)): return False
    spread = np.max(c, axis=0) - np.min(c, axis=0)
    if np.any(spread < 2.0):  return False          # collapsed in one axis
    rms = np.sqrt(np.mean(c**2))
    if rms < 1.0:              return False          # nearly all zeros
    return True


def build_exact_match_index(train_df, train_coords):
    """
    Build O(1) sequence → [(target_id, coords)] dictionary.
    Only includes entries with valid coordinates.
    """
    idx = {}
    for _, row in train_df.iterrows():
        tid = row["target_id"]
        if tid not in train_coords:            continue
        c = train_coords[tid]
        if not _coords_are_valid(c):           continue
        idx.setdefault(row["sequence"], []).append((tid, c))
    n_ent = sum(len(v) for v in idx.values())
    print(f"  Exact-match index: {len(idx)} unique seqs, {n_ent} total entries")
    return idx


def _build_aligned_strings(query_seq, template_seq, alignment):
    q_segs, t_segs = alignment.aligned
    aq, at, qi, ti = [], [], 0, 0
    for (qs,qe),(ts,te) in zip(q_segs, t_segs):
        while qi < qs: aq.append(query_seq[qi]);    at.append("-");              qi += 1
        while ti < ts: aq.append("-");              at.append(template_seq[ti]); ti += 1
        for qp,tp in zip(range(qs,qe),range(ts,te)):
            aq.append(query_seq[qp]); at.append(template_seq[tp])
        qi,ti = qe,te
    while qi < len(query_seq):    aq.append(query_seq[qi]);    at.append("-");              qi += 1
    while ti < len(template_seq): aq.append("-");              at.append(template_seq[ti]); ti += 1
    return "".join(aq), "".join(at)


def find_similar_sequences_detailed(query_seq, train_seqs_df, train_coords_dict,
                                     top_n=30, exact_idx=None):
    """Sequence similarity search with O(1) exact-match fast-path."""
    q_len = len(query_seq)
    exact_results = []
    if exact_idx is not None and query_seq in exact_idx:
        for tid, coords in exact_idx[query_seq]:
            exact_results.append((tid, query_seq, 1.0, coords, 100.0, "", ""))
        if len(exact_results) >= top_n:
            return exact_results[:top_n]
    seen_exact = {r[0] for r in exact_results}

    results = []
    for _, row in train_seqs_df.iterrows():
        tid, tseq = row["target_id"], row["sequence"]
        if tid not in train_coords_dict or tid in seen_exact: continue
        if abs(len(tseq)-q_len)/max(len(tseq),q_len) > 0.3:  continue
        aln    = next(iter(_aligner.align(query_seq, tseq)))
        norm_s = aln.score / (2*min(q_len, len(tseq)))
        identical = sum(
            1 for (qs,qe),(ts,te) in zip(*aln.aligned)
            for qp,tp in zip(range(qs,qe),range(ts,te))
            if query_seq[qp] == tseq[tp]
        )
        pct_id = 100 * identical / q_len
        aq, at = _build_aligned_strings(query_seq, tseq, aln)
        results.append((tid, tseq, norm_s, train_coords_dict[tid], pct_id, aq, at))
    results.sort(key=lambda x: x[2], reverse=True)
    combined = exact_results + results
    seen, final = set(), []
    for item in combined:
        if item[0] not in seen:
            seen.add(item[0]); final.append(item)
    return final[:top_n]


def adapt_template_to_query(query_seq, template_seq, template_coords):
    aln     = next(iter(_aligner.align(query_seq, template_seq)))
    new_c   = np.full((len(query_seq), 3), np.nan)
    for (qs,qe),(ts,te) in zip(*aln.aligned):
        chunk = template_coords[ts:te]
        if len(chunk) == (qe-qs): new_c[qs:qe] = chunk
    for i in range(len(new_c)):
        if np.isnan(new_c[i,0]):
            pv = next((j for j in range(i-1,-1,-1) if not np.isnan(new_c[j,0])), -1)
            nv = next((j for j in range(i+1,len(new_c)) if not np.isnan(new_c[j,0])), -1)
            if   pv>=0 and nv>=0: w=(i-pv)/(nv-pv); new_c[i]=(1-w)*new_c[pv]+w*new_c[nv]
            elif pv>=0:            new_c[i] = new_c[pv]+[3,0,0]
            elif nv>=0:            new_c[i] = new_c[nv]+[3,0,0]
            else:                  new_c[i] = [i*3,0,0]
    return np.nan_to_num(new_c)


def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2):
    """
    RNA geometry constraints.
    ★★★ For exact matches (confidence≥0.99): SKIPPED entirely.
    The minimum strength=0.02 still distorts perfect coordinates.
    """
    if confidence >= 0.99:
        return coords.copy()      # ← exact match: no distortion

    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75*(1.0 - min(confidence, 0.97)), 0.02)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e-s
            if L < 3: continue
            d    = C[1:]-C[:-1]; dist = np.linalg.norm(d, axis=1)+1e-6
            adj  = d*((5.95-dist)/dist)[:,None]*(0.22*strength)
            C[:-1]-=adj; C[1:]+=adj
            d2   = C[2:]-C[:-2]; d2n = np.linalg.norm(d2, axis=1)+1e-6
            adj2 = d2*((10.2-d2n)/d2n)[:,None]*(0.10*strength)
            C[:-2]-=adj2; C[2:]+=adj2
            C[1:-1] += (0.06*strength)*(0.5*(C[:-2]+C[2:])-C[1:-1])
            if L >= 25:
                idx  = np.linspace(0,L-1,min(L,160)).astype(int) if L>220 else np.arange(L)
                P    = C[idx]; diff = P[:,None,:]-P[None,:,:]
                dm   = np.linalg.norm(diff, axis=2)+1e-6
                sep  = np.abs(idx[:,None]-idx[None,:])
                mask = (sep>2)&(dm<3.2)
                if np.any(mask):
                    vec = (diff*((3.2-dm)/dm)[:,:,None]*mask[:,:,None]).sum(axis=1)
                    C[idx] += (0.015*strength)*vec
            X[s:e] = C
    return X


# ── Diversity transforms ──────────────────────────────────────────────────────
def _rotmat(axis, ang):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)+1e-12
    x,y,z = a; c,s = np.cos(ang),np.sin(ang); CC=1-c
    return np.array([[c+x*x*CC, x*y*CC-z*s, x*z*CC+y*s],
                     [y*x*CC+z*s, c+y*y*CC, y*z*CC-x*s],
                     [z*x*CC-y*s, z*y*CC+x*s, c+z*z*CC]])

def apply_hinge(coords, seg, rng, deg=22):
    s,e = seg; L = e-s
    if L < 30: return coords
    pivot = s + int(rng.integers(10, L-10))
    R = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg,deg))))
    X = coords.copy(); p0 = X[pivot].copy()
    X[pivot+1:e] = (X[pivot+1:e]-p0)@R.T+p0
    return X

def apply_double_hinge(coords, seg, rng, deg=15):
    s,e = seg; L = e-s
    if L < 60: return apply_hinge(coords, seg, rng, deg)
    p1 = s + int(rng.integers(10, L//2))
    p2 = s + int(rng.integers(L//2, L-10))
    R1 = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg,deg))))
    R2 = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg,deg))))
    X = coords.copy()
    p0 = X[p1].copy(); X[p1+1:p2] = (X[p1+1:p2]-p0)@R1.T+p0
    p0 = X[p2].copy(); X[p2+1:e]  = (X[p2+1:e]-p0)@R2.T+p0
    return X

def jitter_chains(coords, segs, rng, deg=12, trans=1.5):
    X = coords.copy(); gc_ = X.mean(0, keepdims=True)
    for s,e in segs:
        R     = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg,deg))))
        shift = rng.normal(size=3)
        shift = shift/(np.linalg.norm(shift)+1e-12)*float(rng.uniform(0,trans))
        c = X[s:e].mean(0, keepdims=True)
        X[s:e] = (X[s:e]-c)@R.T+c+shift
    X -= X.mean(0, keepdims=True)-gc_
    return X

def smooth_wiggle(coords, segs, rng, amp=0.8):
    X = coords.copy()
    for s,e in segs:
        L = e-s
        if L < 20: continue
        ctrl = np.linspace(0,L-1,6); disp = rng.normal(0,amp,(6,3)); t = np.arange(L)
        X[s:e] += np.vstack([np.interp(t,ctrl,disp[:,k]) for k in range(3)]).T
    return X

def generate_rna_structure(sequence, seed=None):
    if seed is not None: np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n,3))
    for i in range(n):
        ang = i*0.6; coords[i] = [10.0*np.cos(ang), 10.0*np.sin(ang), i*2.5]
    return coords


# ══════════════════════════════════════════════════════════════════════════════
# TBM PHASE
# ══════════════════════════════════════════════════════════════════════════════
def tbm_phase(test_df, train_seqs_df, train_coords_dict, segments_map, exact_idx=None):
    """
    Phase 1 — Template-Based Modeling.

    Slot assignment:
      slot 0 → best template (no diversity transform)
      slot 1 → softmax-sampled template + small Gaussian jitter
      slot 2 → softmax-sampled template + hinge / double-hinge
      slot 3 → softmax-sampled template + jitter_chains
      slot 4 → softmax-sampled template + smooth_wiggle

    ★★★ For exact-match slot 0 (pct_id≥99): adaptive_rna_constraints SKIPPED.
    ★★  Slots 1-4: softmax diversity sampling from top-15 candidates (τ=0.08),
        down-weighting already-used templates by 0.10×.
    """
    print(f"\n{'='*65}")
    print("PHASE 1: Template-Based Modeling")
    print(f"  MIN_SIMILARITY={MIN_SIMILARITY} | MIN_PCT_IDENTITY={MIN_PERCENT_IDENTITY}")
    print(f"{'='*65}")
    t0 = time.time()
    template_predictions, protenix_queue = {}, {}

    for _, row in test_df.iterrows():
        tid  = row["target_id"]
        seq  = row["sequence"]
        segs = segments_map.get(tid, [(0, len(seq))])

        similar = find_similar_sequences_detailed(
            seq, train_seqs_df, train_coords_dict, top_n=30, exact_idx=exact_idx
        )

        preds, used = [], set()

        # ★★★ v19 SEQUENTIAL BREAK (reference approach):
        # Iterate sorted-by-sim candidates in order; break at first failing threshold.
        # This naturally routes hard targets (9JFS, 9LEC, 9LEL) to Protenix when
        # their best templates fall below MIN_PERCENT_IDENTITY.
        # Pre-filtering was the v17/v18 bug: it found all pct_id≥50% templates
        # across top_n=50, filling every slot and preventing Protenix from running.
        for tmpl_id, tmpl_seq, sim, tmpl_coords, pct_id, _, _ in similar:
            if len(preds) >= N_SAMPLE:
                break
            # ★★★ BREAK (not continue) — list sorted by sim; first failure means done
            if sim < MIN_SIMILARITY or pct_id < MIN_PERCENT_IDENTITY:
                break
            if tmpl_id in used:
                continue

            slot    = len(preds)
            rng     = np.random.default_rng((row.name*10_000_000_000 + slot*10007) % (2**32))
            adapted = adapt_template_to_query(seq, tmpl_seq, tmpl_coords)
            longest = max(segs, key=lambda se: se[1]-se[0])
            longest_len = longest[1] - longest[0]

            if slot == 0:
                # Slot 0: best template, apply constraints for geometry cleanup
                X = adaptive_rna_constraints(adapted, tid, segments_map, confidence=sim)
            elif slot == 1:
                # Slot 1: gentle Gaussian jitter (halved for near-perfect templates)
                noise_scale = max(0.005, (0.40-sim)*0.03) if pct_id > 95 else max(0.01,(0.40-sim)*0.06)
                X = adapted + rng.normal(0, noise_scale, adapted.shape)
                X = adaptive_rna_constraints(X, tid, segments_map, confidence=sim)
            elif slot == 2:
                # Slot 2: double-hinge for long segs (≥100 nt), single-hinge for short
                X = apply_double_hinge(adapted, longest, rng) if longest_len >= 100 else apply_hinge(adapted, longest, rng)
                X = adaptive_rna_constraints(X, tid, segments_map, confidence=sim)
            elif slot == 3:
                # Slot 3: chain jitter (rigid-body per-chain rotation+translation)
                X = jitter_chains(adapted, segs, rng)
                X = adaptive_rna_constraints(X, tid, segments_map, confidence=sim)
            else:
                # Slot 4: smooth wavelike displacement
                X = smooth_wiggle(adapted, segs, rng)
                X = adaptive_rna_constraints(X, tid, segments_map, confidence=sim)

            preds.append(X)
            used.add(tmpl_id)

        template_predictions[tid] = preds
        n_needed = N_SAMPLE - len(preds)
        if n_needed > 0:
            protenix_queue[tid] = (n_needed, seq)
            print(f"  {tid} ({len(seq)} nt): {len(preds)} TBM → need {n_needed} Protenix")
        else:
            print(f"  {tid} ({len(seq)} nt): all {N_SAMPLE} TBM ✓")

    elapsed = time.time()-t0
    print(f"\nPhase 1 done in {elapsed:.1f}s | "
          f"TBM-only:{len(test_df)-len(protenix_queue)} | Protenix:{len(protenix_queue)}")
    return template_predictions, protenix_queue


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — PROTENIX  (single GPU, main process, chunking + Kabsch)
# ══════════════════════════════════════════════════════════════════════════════
def run_protenix(protenix_queue, work_dir, code_dir, root_dir, seed=None):
    """
    Phase 2 — Protenix neural network inference.

    • Runs entirely in the main process on GPU 0 → full tracebacks visible.
    • Long sequences (>MAX_SEQ_LEN) are split into overlapping chunks.
    • Chunk outputs are aligned via Kabsch and blended with linear weight ramps.
    • C1' atoms selected via atom_to_tokatom_idx (tries idx=12 first, then idx=11,
      picking whichever is closest to the target sequence length).
    • v17: accepts seed parameter for dual-seed diversity runs.
    """
    print(f"\n{'='*65}")
    print(f"PHASE 2: Protenix Inference (single GPU, chunked for long seqs, seed={seed or SEED})")
    print(f"{'='*65}")

    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        try:
            free, total = torch.cuda.mem_get_info(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                  f"({free/1024**3:.1f}/{total/1024**3:.1f} GB free)")
        except:
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # ── 1. Build task list (with chunk info for long seqs) ────────────────────
    tasks, chunk_info = [], {}
    for target_id, (n_needed, full_seq) in protenix_queue.items():
        seq_len = len(full_seq)
        if seq_len <= MAX_SEQ_LEN:
            tasks.append({"target_id": target_id, "sequence": full_seq})
            chunk_info[target_id] = [{"name": target_id, "range": (0, seq_len)}]
            print(f"  {target_id} ({seq_len} nt): single pass")
        else:
            chunks = split_into_chunks(seq_len, MAX_SEQ_LEN, CHUNK_OVERLAP)
            print(f"  {target_id} ({seq_len} nt): {len(chunks)} chunks")
            chunk_info[target_id] = []
            for ci, (cs, ce) in enumerate(chunks):
                cn = f"{target_id}_chunk{ci}"
                tasks.append({"target_id": cn, "sequence": full_seq[cs:ce]})
                chunk_info[target_id].append({"name": cn, "range": (cs, ce)})

    tasks_df        = pd.DataFrame(tasks)
    input_json_path = str(work_dir / "protenix_input.json")
    build_input_json(tasks_df, input_json_path)

    # ── 2. Initialize Protenix runner & dataset ───────────────────────────────
    from protenix.data.inference.infer_dataloader import InferenceDataset
    from runner.inference import (InferenceRunner,
                                  update_gpu_compatible_configs,
                                  update_inference_configs)

    configs = build_configs(input_json_path, str(work_dir/"outputs"), MODEL_NAME, seed=seed)
    configs = update_gpu_compatible_configs(configs)
    runner  = InferenceRunner(configs)
    dataset = InferenceDataset(configs)

    # ── 3. Inference loop ─────────────────────────────────────────────────────
    raw_predictions = {}   # sample_name → np.ndarray(n_needed, seq_len, 3) | None

    def _get_c1_mask(data, atom_array, chunk_seq_len):
        """
        v18: Reference-quality C1 mask selection.
        Priority: atom_array attributes -> feature dict -> tokatom_idx heuristic.
        """
        if atom_array is not None:
            try:
                if hasattr(atom_array, "centre_atom_mask"):
                    m = atom_array.centre_atom_mask == 1
                    if hasattr(atom_array, "is_rna"):
                        m = m & atom_array.is_rna
                    return torch.from_numpy(m).bool()
                if hasattr(atom_array, "atom_name"):
                    base = atom_array.atom_name == "C1'"
                    if hasattr(atom_array, "is_rna"):
                        base = base & atom_array.is_rna
                    return torch.from_numpy(base).bool()
            except Exception:
                pass
        f = data["input_feature_dict"]
        if "centre_atom_mask" in f:
            return (f["centre_atom_mask"] == 1).bool()
        if "center_atom_mask" in f:
            return (f["center_atom_mask"] == 1).bool()
        m11 = (f["atom_to_tokatom_idx"] == 11).bool()
        m12 = (f["atom_to_tokatom_idx"] == 12).bool()
        c11, c12 = m11.sum().item(), m12.sum().item()
        return m11 if abs(c11 - chunk_seq_len) < abs(c12 - chunk_seq_len) else m12

    def _extract_c1_coords(pred, data, atom_array, chunk_seq_len, raw_coords):
        """Select C1' atom coordinates from Protenix output using reference logic."""
        mask = _get_c1_mask(data, atom_array, chunk_seq_len).to(raw_coords.device)
        coords = raw_coords[:, mask, :].detach().cpu().numpy()
        if coords.shape[1] > 1:
            diffs = np.linalg.norm(coords[0,1:]-coords[0,:-1], axis=-1)
            if np.all(diffs < 1e-4):
                print(f"    WARNING: collapsed coordinates detected"); return None
        if coords.shape[1] != chunk_seq_len:
            if coords.shape[1] == 1 and chunk_seq_len > 1: return None
            padded = np.zeros((coords.shape[0], chunk_seq_len, 3), dtype=np.float32)
            ml = min(coords.shape[1], chunk_seq_len)
            padded[:, :ml, :] = coords[:, :ml, :]
            coords = padded
        return coords

    for i in tqdm(range(len(dataset)), desc="Protenix"):
        data, atom_array, err = dataset[i]
        sample_name = data.get("sample_name", f"sample_{i}")

        if err:
            print(f"  {sample_name}: data error — {err}")
            raw_predictions[sample_name] = None
            del data, atom_array, err
            gc.collect(); torch.cuda.empty_cache(); gc.collect()
            continue

        target_id   = sample_name.split("_chunk")[0] if "_chunk" in sample_name else sample_name
        n_needed    = protenix_queue.get(target_id, (N_SAMPLE,""))[0]
        sub_seq_len = data["N_token"].item()

        try:
            new_cfg = update_inference_configs(configs, sub_seq_len)
            new_cfg.sample_diffusion.N_sample = n_needed
            runner.update_model_configs(new_cfg)
            pred       = runner.predict(data)
            raw_coords = pred["coordinate"]
            coords     = _extract_c1_coords(pred, data, atom_array,
                                             sub_seq_len, raw_coords)
            raw_predictions[sample_name] = coords
            if coords is not None:
                print(f"\n  {sample_name}: {coords.shape[0]} preds ✓")
            else:
                print(f"\n  {sample_name}: extraction failed")
        except Exception as exc:
            import traceback
            print(f"\n  {sample_name}: FAILED — {exc}")
            traceback.print_exc()
            raw_predictions[sample_name] = None
        finally:
            try: del pred, data, atom_array, raw_coords
            except: pass
            gc.collect(); torch.cuda.empty_cache(); gc.collect()

    # ── 4. Stitch chunks → protenix_preds ────────────────────────────────────
    protenix_preds = {}
    for target_id, (n_needed, full_seq) in protenix_queue.items():
        seq_len = len(full_seq)
        chunks  = chunk_info.get(target_id, [])
        if not chunks: continue

        if len(chunks) == 1:
            coords = raw_predictions.get(target_id)
            protenix_preds[target_id] = coords
            if coords is not None:
                print(f"  {target_id}: {coords.shape[0]} preds ✓")
            else:
                print(f"  {target_id}: FAILED → fallback to TBM / de-novo")
        else:
            per_sample = {s: [] for s in range(n_needed)}
            all_ok     = True
            for cinfo in chunks:
                ccoords = raw_predictions.get(cinfo["name"])
                if ccoords is None:
                    all_ok = False; break
                for s_idx in range(n_needed):
                    si = s_idx if s_idx < ccoords.shape[0] else -1
                    per_sample[s_idx].append((ccoords[si], cinfo["range"]))
            if not all_ok:
                print(f"  {target_id}: chunked incomplete → fallback")
                protenix_preds[target_id] = None; continue
            stitched = []
            for s_idx in range(n_needed):
                items = per_sample[s_idx]
                fc = stitch_chunk_coords([c for c,_ in items],
                                          [r for _,r in items], seq_len)
                stitched.append(fc)
            result = np.stack(stitched, axis=0)
            protenix_preds[target_id] = result
            print(f"  {target_id}: {result.shape[0]} stitched preds ✓")

    return protenix_preds


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t_total = time.time()
    test_csv, output_csv, code_dir, root_dir = resolve_paths()

    if not os.path.isdir(code_dir):
        raise FileNotFoundError(f"Missing PROTENIX_CODE_DIR: {code_dir}")

    os.environ["PROTENIX_ROOT_DIR"] = root_dir
    sys.path.append(code_dir)
    ensure_required_files(root_dir)
    seed_everything(SEED)

    print(f"GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                  f"({free/1024**3:.1f}/{total/1024**3:.1f} GB free)")
        except:
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # ── Load data ──────────────────────────────────────────────────────────────
    test_df = pd.read_csv(test_csv).reset_index(drop=True)
    print(f"\nTest targets: {len(test_df)}")

    print("\nLoading training + validation data…")
    train_seqs   = pd.read_csv(DEFAULT_TRAIN_CSV)
    val_seqs     = pd.read_csv(DEFAULT_VAL_CSV)
    train_labels = pd.read_csv(DEFAULT_TRAIN_LBLS, low_memory=False)
    val_labels   = pd.read_csv(DEFAULT_VAL_LBLS)

    combined_seqs   = pd.concat([train_seqs,   val_seqs],   ignore_index=True)
    combined_labels = pd.concat([train_labels, val_labels], ignore_index=True)

    del train_seqs, val_seqs, train_labels, val_labels
    gc.collect()

    # ★★★ ID-suffix sort fix
    print("Processing labels (★★★ ID-suffix sort fix)…")
    train_coords = process_labels(combined_labels)
    del combined_labels
    gc.collect()

    segments_map, _ = build_segments_map(test_df)
    print(f"Template pool: {len(combined_seqs)} seqs, {len(train_coords)} structures")

    # ★★ Exact-match O(1) index
    exact_idx = build_exact_match_index(combined_seqs, train_coords)

    # ── PHASE 1: TBM ──────────────────────────────────────────────────────────
    template_preds, protenix_queue = tbm_phase(
        test_df, combined_seqs, train_coords, segments_map, exact_idx=exact_idx
    )

    # ── PHASE 2: Protenix ─────────────────────────────────────────────────────
    # v17: dual-seed strategy — run seed=101, then seed=202 for hard targets
    # Hard targets: those that got 0 TBM predictions (fully reliant on Protenix)
    protenix_preds = {}
    if protenix_queue and USE_PROTENIX:
        work_dir = Path("/kaggle/working")
        work_dir.mkdir(parents=True, exist_ok=True)

        # Identify hard targets (0 TBM preds) for dual-seed
        hard_targets  = {tid for tid, preds in template_preds.items() if len(preds) == 0}
        easy_queue    = {tid: v for tid, v in protenix_queue.items() if tid not in hard_targets}
        hard_queue    = {tid: v for tid, v in protenix_queue.items() if tid in hard_targets}

        print(f"\n  v17 Protenix: {len(easy_queue)} easy (single seed), "
              f"{len(hard_queue)} hard (dual seed)")

        # Seed 1: run all targets with SEED=101
        protenix_preds = run_protenix(protenix_queue, work_dir, code_dir, root_dir, seed=101)

        # Seed 2: for hard targets only, run again with SEED=202
        if hard_queue:
            print(f"\n  v17 Dual-seed pass for {len(hard_queue)} hard targets (seed=202)…")
            preds_seed2 = run_protenix(hard_queue, work_dir, code_dir, root_dir, seed=202)
            # For hard targets, replace single-seed result with seed-1 + seed-2 combined
            # We keep seed-1 preds in slots 0..n1-1, seed-2 preds in slots n1..N_SAMPLE-1
            for tid, (n_needed, full_seq) in hard_queue.items():
                p1 = protenix_preds.get(tid)
                p2 = preds_seed2.get(tid)
                if p1 is not None and p2 is not None and p1.ndim == 3 and p2.ndim == 3:
                    # Take min(n1, ceil(N/2)) from each seed for balanced diversity
                    half = N_SAMPLE // 2
                    take1 = min(p1.shape[0], N_SAMPLE - half)
                    take2 = min(p2.shape[0], half)
                    combined_ptx = np.concatenate([p1[:take1], p2[:take2]], axis=0)
                    protenix_preds[tid] = combined_ptx
                    print(f"    {tid}: dual-seed combined "
                          f"({take1} seed-101 + {take2} seed-202)")
                elif p1 is None and p2 is not None:
                    protenix_preds[tid] = p2  # fallback to seed-2
    elif protenix_queue and not USE_PROTENIX:
        print(f"\nPHASE 2 skipped (USE_PROTENIX=False).")

    # ── PHASE 3: Combine TBM + Protenix + de-novo ────────────────────────────
    print(f"\n{'='*65}")
    print("PHASE 3: Combine TBM + Protenix + de-novo fallback")
    print(f"{'='*65}")
    all_rows = []

    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]
        combined = list(template_preds.get(tid, []))
        n_tbm = len(combined)

        # Append Protenix predictions (v17: no constraints — Protenix geometry is already valid)
        ptx = protenix_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE: break
                # v17: use raw Protenix output — constraints degrade neural-net geometry
                combined.append(ptx[j].astype(np.float64))
        n_ptx = len(combined) - n_tbm

        # De-novo fallback for remaining slots
        n_dn = 0
        while len(combined) < N_SAMPLE:
            seed_val = row.name*1_000_000 + len(combined)*1000
            dn = generate_rna_structure(seq, seed=seed_val)
            combined.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.2))
            n_dn += 1

        # v17: diagnostic — show strategy for every target
        strategy = f"TBM={n_tbm} PTX={n_ptx}"
        if n_dn: strategy += f" DN={n_dn}"
        print(f"  {tid} ({len(seq)} nt): {strategy}")

        stacked = np.stack(combined[:N_SAMPLE], axis=0)
        all_rows.extend(coords_to_rows(tid, seq, stacked))

    # ── Save submission ────────────────────────────────────────────────────────
    sub = pd.DataFrame(all_rows)
    cols = ["ID","resname","resid"] + [
        f"{c}_{i}" for i in range(1,N_SAMPLE+1) for c in ["x","y","z"]
    ]
    cc = [c for c in cols if c.startswith(("x_","y_","z_"))]
    sub[cc] = sub[cc].clip(-999.999, 9999.999)
    sub[cols].to_csv(output_csv, index=False)

    elapsed = time.time()-t_total
    print(f"\n✓ Saved {output_csv} ({len(sub):,} rows) in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
