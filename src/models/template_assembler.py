from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.folding_transformer import SE3RefinementBlock


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner = dim * mult
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class LocalGraphMessageBlock(nn.Module):
    """
    Local graph block over residues inside a chunk.
    Operates independently per candidate chunk:
      complexity ~ O(B * W * K * C * |offsets|)
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        offsets: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.offsets = tuple(sorted({int(abs(o)) for o in offsets if int(abs(o)) > 0}))
        if len(self.offsets) == 0:
            self.offsets = (1,)

        self.node_norm = nn.LayerNorm(hidden_dim)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.update = ResidualMLP(hidden_dim, mult=4, dropout=dropout)

    def _edge_features(
        self,
        xyz_i: torch.Tensor,
        xyz_j: torch.Tensor,
    ) -> torch.Tensor:
        delta = xyz_j - xyz_i
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp(min=1e-6)
        unit = delta / dist
        return torch.cat([unit, dist], dim=-1)  # (..., 4)

    def forward(
        self,
        h: torch.Tensor,      # (N, C, H)
        xyz: torch.Tensor,    # (N, C, 3)
        valid: torch.Tensor,  # (N, C)
    ) -> torch.Tensor:
        n, c, hd = h.shape
        assert hd == self.hidden_dim

        h_in = self.node_norm(h)
        agg = torch.zeros_like(h_in)

        for off in self.offsets:
            if off >= c:
                continue

            src = slice(off, c)
            dst = slice(0, c - off)
            v = valid[:, src] & valid[:, dst]
            if not bool(v.any().item()):
                continue

            h_i = h_in[:, dst]
            h_j = h_in[:, src]

            e_fwd = self._edge_features(xyz[:, dst], xyz[:, src])
            msg_fwd = self.msg_mlp(torch.cat([h_i, h_j, e_fwd], dim=-1))
            agg[:, dst] = agg[:, dst] + msg_fwd * v.unsqueeze(-1).float()

            e_rev = self._edge_features(xyz[:, src], xyz[:, dst])
            msg_rev = self.msg_mlp(torch.cat([h_j, h_i, e_rev], dim=-1))
            agg[:, src] = agg[:, src] + msg_rev * v.unsqueeze(-1).float()

        h = h + agg * valid.unsqueeze(-1).float()
        h = self.update(h) * valid.unsqueeze(-1).float()
        return h


class TemplateSegmentAssembler(nn.Module):
    """
    Mixture-of-templates RNA assembler.

    Pipeline:
    1) Per-candidate local graph encoder over chunk residues.
    2) Global sequence-context transformer over absolute target positions.
    3) Shared 2-layer scorer over [reference + all candidates]:
       - token-level candidate weights
       - window-level candidate weights
    4) Sparse confidence-weighted mixture assembly.
    5) Seam-aware weighted Kabsch stitching.
    6) Optional refinement:
       - transformer, or
       - SE(3) equivariant refinement
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_candidates: int = 20,
        residue_vocab_size: int = 5,
        max_chain_embeddings: int = 64,
        max_copy_embeddings: int = 64,
        template_chunk_length: int = 512,
        template_chunk_stride: int = 256,
        template_chunk_max_windows: int = 64,
        seq_transformer_layers: int = 4,
        seq_transformer_heads: int = 8,
        transformer_dropout: float = 0.0,
        cross_attention_heads: int = 8,
        sparse_window: int = 256,
        sparse_mixture_topk: int = 2,
        graph_layers: int = 3,
        graph_offsets: Sequence[int] = (1, 2, 4, 8, 16),
        rigid_alignment_enabled: bool = True,
        refinement_block: str = "se3",
        se3_refinement_enabled: bool = True,
        se3_refine_layers: int = 4,
        se3_num_heads: int = 4,
        se3_dropout: float = 0.0,
        se3_coord_step_size: float = 0.1,
        geometric_constraints_enabled: bool = False,
        geometric_constraints_strength: float = 0.75,
        geometric_constraints_nonlocal_max_points: int = 160,
        geometric_constraints_nonlocal_min_sep: int = 2,
        bond_target: float = 5.95,
        twohop_target: float = 10.2,
        clash_min_distance: float = 3.2,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_candidates = max(1, int(num_candidates))
        self.template_chunk_length = max(1, int(template_chunk_length))
        self.template_chunk_stride = max(1, int(template_chunk_stride))
        self.template_chunk_max_windows = max(1, int(template_chunk_max_windows))
        self.max_sequence_positions = self.template_chunk_length + self.template_chunk_stride * max(
            0, self.template_chunk_max_windows - 1
        )

        self.refinement_block = str(refinement_block).strip().lower()
        if self.refinement_block not in {"transformer", "se3"}:
            raise ValueError("refinement_block must be one of {'transformer', 'se3'}")

        self.se3_refinement_enabled = bool(se3_refinement_enabled)
        self.rigid_alignment_enabled = bool(rigid_alignment_enabled)
        self.sparse_mixture_topk = max(1, int(sparse_mixture_topk))

        self.use_geom_aux = bool(geometric_constraints_enabled)
        self.geom_aux_weight = float(geometric_constraints_strength)
        self.geom_aux_max_points = max(16, int(geometric_constraints_nonlocal_max_points))
        self.geom_aux_min_sep = max(0, int(geometric_constraints_nonlocal_min_sep))
        self.bond_target = float(bond_target)
        self.twohop_target = float(twohop_target)
        self.clash_min_distance = float(clash_min_distance)

        # Target sequence features.
        self.residue_emb = nn.Embedding(residue_vocab_size, hidden_dim)
        self.chain_emb = nn.Embedding(max_chain_embeddings, hidden_dim)
        self.copy_emb = nn.Embedding(max_copy_embeddings, hidden_dim)
        self.resid_proj = nn.Linear(1, hidden_dim)

        # Candidate chunk features.
        self.chunk_residue_emb = nn.Embedding(residue_vocab_size, hidden_dim)
        self.chunk_pos_emb = nn.Embedding(self.template_chunk_length, hidden_dim)
        self.window_emb = nn.Embedding(self.template_chunk_max_windows, hidden_dim)
        self.chunk_coord_proj = nn.Linear(3, hidden_dim)

        # identity, similarity, confidence, source0, source1, valid, coverage
        self.match_meta_proj = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Local graph encoder.
        self.graph_encoder = nn.ModuleList(
            [
                LocalGraphMessageBlock(
                    hidden_dim=hidden_dim,
                    edge_dim=4,
                    offsets=graph_offsets,
                    dropout=float(transformer_dropout),
                )
                for _ in range(max(1, int(graph_layers)))
            ]
        )

        # Scoring context.
        self.score_role_emb = nn.Embedding(max(16, self.num_candidates + 1), hidden_dim)
        self.score_norm = nn.LayerNorm(hidden_dim)
        self.score_global_norm = nn.LayerNorm(hidden_dim)

        self.score_global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=max(1, int(seq_transformer_heads)),
                dim_feedforward=hidden_dim * 4,
                dropout=float(transformer_dropout),
                activation="gelu",
                norm_first=True,
                batch_first=True,
            ),
            num_layers=2,
            enable_nested_tensor=False,
        )

        self.score_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=max(1, int(cross_attention_heads)),
                dim_feedforward=hidden_dim * 4,
                dropout=float(transformer_dropout),
                activation="gelu",
                norm_first=True,
                batch_first=True,
            ),
            num_layers=2,
            enable_nested_tensor=False,
        )

        self.token_score_head = nn.Linear(hidden_dim, 1)
        self.window_score_head = nn.Linear(hidden_dim, 1)

        # Seam profile.
        self.seam_logits = nn.Parameter(torch.zeros(self.template_chunk_length))

        # Refinement.
        self.coord_context_proj = nn.Linear(3, hidden_dim)
        self.conf_context_proj = nn.Linear(1, hidden_dim)
        self.refine_norm = nn.LayerNorm(hidden_dim)

        self.conf_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        if self.refinement_block == "transformer":
            self.global_layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=hidden_dim,
                        nhead=max(1, int(seq_transformer_heads)),
                        dim_feedforward=hidden_dim * 4,
                        dropout=float(transformer_dropout),
                        activation="gelu",
                        norm_first=True,
                        batch_first=True,
                    )
                    for _ in range(max(1, int(seq_transformer_layers)))
                ]
            )
            self.coord_delta_head: Optional[nn.Linear] = nn.Linear(hidden_dim, 3)
            self.coord_to_hidden: Optional[nn.Linear] = None
            self.se3_layers = nn.ModuleList()
        else:
            self.global_layers = nn.ModuleList()
            self.coord_delta_head = None
            if self.se3_refinement_enabled:
                self.coord_to_hidden = nn.Linear(3, hidden_dim)
                self.se3_layers = nn.ModuleList(
                    [
                        SE3RefinementBlock(
                            hidden_dim=hidden_dim,
                            num_heads=max(1, int(se3_num_heads)),
                            dropout=float(se3_dropout),
                            coord_step_size=float(se3_coord_step_size),
                            use_fast_multipole=False,
                            use_block_sparse_attention=True,
                            block_sparse_min_seq_len=1,
                            block_sparse_block_size=64,
                            block_sparse_window=max(1, int(sparse_window)),
                            block_sparse_global_stride=128,
                            block_sparse_max_global=64,
                            block_sparse_geo_topk=16,
                            pure_sparse_mode=True,
                        )
                        for _ in range(max(1, int(se3_refine_layers)))
                    ]
                )
            else:
                self.coord_to_hidden = None
                self.se3_layers = nn.ModuleList()

        self.last_aux_losses: dict[str, torch.Tensor] = {}
        self.last_confidence: Optional[torch.Tensor] = None
        self.last_overlap_compatibility: Optional[torch.Tensor] = None
        self.last_window_candidate_scores: Optional[torch.Tensor] = None
        self.last_token_candidate_scores: Optional[torch.Tensor] = None

    def _prepare_chunk_inputs(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: torch.Tensor,
        template_chunk_coords: Optional[torch.Tensor],
        template_chunk_mask: Optional[torch.Tensor],
        template_chunk_start: Optional[torch.Tensor],
        template_chunk_window_valid: Optional[torch.Tensor],
        template_chunk_valid: Optional[torch.Tensor],
        template_chunk_identity: Optional[torch.Tensor],
        template_chunk_similarity: Optional[torch.Tensor],
        template_chunk_confidence: Optional[torch.Tensor],
        template_chunk_source_onehot: Optional[torch.Tensor],
        template_chunk_residue_idx: Optional[torch.Tensor],
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        b, l, _ = coords.shape
        if template_chunk_coords is None or template_chunk_coords.dim() != 5:
            c = torch.zeros(
                b, 1, self.num_candidates, self.template_chunk_length, 3,
                device=coords.device, dtype=coords.dtype,
            )
            chunk_mask = torch.zeros(b, 1, self.template_chunk_length, device=coords.device, dtype=torch.bool)
            chunk_start = torch.zeros(b, 1, device=coords.device, dtype=torch.long)
            chunk_window_valid = torch.zeros(b, 1, device=coords.device, dtype=torch.bool)
            chunk_valid = torch.zeros(b, 1, self.num_candidates, device=coords.device, dtype=torch.bool)
            chunk_identity = torch.zeros(b, 1, self.num_candidates, device=coords.device, dtype=coords.dtype)
            chunk_similarity = torch.zeros(b, 1, self.num_candidates, device=coords.device, dtype=coords.dtype)
            chunk_conf = torch.zeros(b, 1, self.num_candidates, device=coords.device, dtype=coords.dtype)
            chunk_source = torch.zeros(b, 1, self.num_candidates, 2, device=coords.device, dtype=coords.dtype)
            chunk_residue = torch.full(
                (b, 1, self.num_candidates, self.template_chunk_length),
                4, device=coords.device, dtype=torch.long,
            )
            for bi in range(b):
                take = min(self.template_chunk_length, int(mask[bi].sum().item()), l)
                if take <= 0:
                    continue
                c[bi, 0, 0, :take] = coords[bi, :take]
                chunk_mask[bi, 0, :take] = True
                chunk_window_valid[bi, 0] = True
                chunk_valid[bi, 0, 0] = True
                chunk_identity[bi, 0, 0] = 100.0
                chunk_similarity[bi, 0, 0] = 1.0
                chunk_source[bi, 0, 0, 0] = 1.0
                chunk_residue[bi, 0, 0, :take] = residue_idx[bi, :take]
            return (
                c,
                chunk_mask,
                chunk_start,
                chunk_window_valid,
                chunk_valid,
                chunk_identity,
                chunk_similarity,
                chunk_conf,
                chunk_source,
                chunk_residue,
            )

        c = template_chunk_coords.to(device=coords.device, dtype=coords.dtype)
        w_take = min(c.shape[1], self.template_chunk_max_windows)
        k_take = min(c.shape[2], self.num_candidates)
        c = c[:, :w_take, :k_take]
        _, w, k, clen, _ = c.shape

        chunk_mask = (
            torch.ones(b, w, clen, device=coords.device, dtype=torch.bool)
            if template_chunk_mask is None
            else template_chunk_mask[:, :w, :clen].to(device=coords.device, dtype=torch.bool)
        )
        chunk_start = (
            torch.arange(w, device=coords.device, dtype=torch.long).unsqueeze(0).expand(b, -1) * self.template_chunk_stride
            if template_chunk_start is None
            else template_chunk_start[:, :w].to(device=coords.device, dtype=torch.long)
        )
        chunk_window_valid = (
            chunk_mask.any(dim=-1)
            if template_chunk_window_valid is None
            else template_chunk_window_valid[:, :w].to(device=coords.device, dtype=torch.bool)
        )
        chunk_valid = (
            torch.ones(b, w, k, device=coords.device, dtype=torch.bool)
            if template_chunk_valid is None
            else template_chunk_valid[:, :w, :k].to(device=coords.device, dtype=torch.bool)
        )
        chunk_identity = (
            torch.zeros(b, w, k, device=coords.device, dtype=coords.dtype)
            if template_chunk_identity is None
            else template_chunk_identity[:, :w, :k].to(device=coords.device, dtype=coords.dtype)
        )
        chunk_similarity = (
            torch.zeros(b, w, k, device=coords.device, dtype=coords.dtype)
            if template_chunk_similarity is None
            else template_chunk_similarity[:, :w, :k].to(device=coords.device, dtype=coords.dtype)
        )
        chunk_conf = (
            torch.zeros(b, w, k, device=coords.device, dtype=coords.dtype)
            if template_chunk_confidence is None
            else template_chunk_confidence[:, :w, :k].to(device=coords.device, dtype=coords.dtype)
        )
        if template_chunk_source_onehot is None:
            chunk_source = torch.zeros(b, w, k, 2, device=coords.device, dtype=coords.dtype)
            chunk_source[..., 0] = 1.0
        else:
            chunk_source = template_chunk_source_onehot[:, :w, :k, :2].to(device=coords.device, dtype=coords.dtype)
        if template_chunk_residue_idx is None:
            chunk_residue = torch.full((b, w, k, clen), 4, device=coords.device, dtype=torch.long)
        else:
            chunk_residue = template_chunk_residue_idx[:, :w, :k, :clen].to(device=coords.device, dtype=torch.long)

        return (
            c,
            chunk_mask,
            chunk_start,
            chunk_window_valid,
            chunk_valid,
            chunk_identity,
            chunk_similarity,
            chunk_conf,
            chunk_source,
            chunk_residue,
        )

    def _normalize_candidate_probs(self, logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        logits = logits.masked_fill(~valid, -1e4)
        probs = torch.softmax(logits, dim=-1) * valid.float()
        denom = probs.sum(dim=-1, keepdim=True)
        fallback = valid.float()
        fallback_denom = fallback.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return torch.where(denom > 1e-6, probs / denom.clamp(min=1e-6), fallback / fallback_denom)

    def _sparsify_candidate_probs(self, probs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        k_total = int(probs.shape[-1])
        k_keep = min(max(1, int(self.sparse_mixture_topk)), k_total)
        full = self._normalize_candidate_probs(torch.log(probs.clamp(min=1e-8)), valid)
        if k_keep >= k_total:
            return full
        masked = full.masked_fill(~valid, -1.0)
        top_idx = torch.topk(masked, k=k_keep, dim=-1).indices
        keep = torch.zeros_like(valid)
        keep.scatter_(-1, top_idx, True)
        keep = keep & valid
        sparse = full * keep.float()
        sparse_denom = sparse.sum(dim=-1, keepdim=True)
        return torch.where(sparse_denom > 1e-6, sparse / sparse_denom.clamp(min=1e-6), full)

    def _weighted_kabsch_transform(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = source.device
        dtype = source.dtype
        eye = torch.eye(3, device=device, dtype=dtype)
        zero = torch.zeros(3, device=device, dtype=dtype)

        if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape or source.shape[-1] != 3:
            return eye, zero
        if int(source.shape[0]) < 3:
            return eye, zero

        w = weights.reshape(-1, 1).to(device=device, dtype=torch.float32).clamp(min=0.0)
        w_sum = w.sum()
        if not bool((w_sum > 1e-6).item()):
            return eye, zero

        src = source.to(dtype=torch.float32)
        tgt = target.to(dtype=torch.float32)
        mu_src = (w * src).sum(dim=0) / w_sum
        mu_tgt = (w * tgt).sum(dim=0) / w_sum
        src_c = src - mu_src
        tgt_c = tgt - mu_tgt
        cov = (w * src_c).transpose(0, 1) @ tgt_c

        try:
            u, _, vh = torch.linalg.svd(cov, full_matrices=False)
        except Exception:
            return eye, zero

        r = vh.transpose(0, 1) @ u.transpose(0, 1)
        if bool((torch.det(r) < 0).item()):
            vh_fix = vh.clone()
            vh_fix[-1, :] *= -1.0
            r = vh_fix.transpose(0, 1) @ u.transpose(0, 1)

        t = mu_tgt - (mu_src @ r.transpose(0, 1))
        return r.to(dtype=dtype), t.to(dtype=dtype)

    def _compute_geom_aux(self, pred: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        device = pred.device
        dtype = pred.dtype
        out: dict[str, torch.Tensor] = {}
        if pred.shape[1] >= 2:
            v = mask[:, 1:] & mask[:, :-1]
            d = torch.linalg.norm(pred[:, 1:] - pred[:, :-1], dim=-1)
            out["bond_length"] = (((d - self.bond_target) ** 2) * v.float()).sum() / v.float().sum().clamp(min=1.0)
        else:
            out["bond_length"] = torch.zeros((), device=device, dtype=dtype)
        if pred.shape[1] >= 3:
            v = mask[:, 2:] & mask[:, :-2]
            d = torch.linalg.norm(pred[:, 2:] - pred[:, :-2], dim=-1)
            out["twohop_length"] = (((d - self.twohop_target) ** 2) * v.float()).sum() / v.float().sum().clamp(min=1.0)
        else:
            out["twohop_length"] = torch.zeros((), device=device, dtype=dtype)
        clash = []
        for b_idx in range(pred.shape[0]):
            idx = torch.nonzero(mask[b_idx], as_tuple=False).squeeze(-1)
            if idx.numel() < 4:
                continue
            if idx.numel() > self.geom_aux_max_points:
                keep = torch.linspace(0, idx.numel() - 1, steps=self.geom_aux_max_points, device=device).round().long()
                idx = idx[keep]
            p = pred[b_idx, idx]
            dist = torch.linalg.norm(p[:, None, :] - p[None, :, :], dim=-1).clamp(min=1e-6)
            sep = (idx[:, None] - idx[None, :]).abs()
            v = sep > self.geom_aux_min_sep
            clash.append((torch.relu(self.clash_min_distance - dist) * v.float()).sum() / v.float().sum().clamp(min=1.0))
        out["clash"] = torch.stack(clash).mean() if clash else torch.zeros((), device=device, dtype=dtype)
        out["total"] = (out["bond_length"] + out["twohop_length"] + out["clash"]) * self.geom_aux_weight
        return out

    def get_aux_losses(self) -> dict[str, torch.Tensor]:
        return self.last_aux_losses

    def forward(
        self,
        residue_idx: torch.Tensor,
        chain_idx: torch.Tensor,
        copy_idx: torch.Tensor,
        resid: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        template_coords: Optional[torch.Tensor] = None,
        template_mask: Optional[torch.Tensor] = None,
        template_topk_coords: Optional[torch.Tensor] = None,
        template_topk_mask: Optional[torch.Tensor] = None,
        template_topk_valid: Optional[torch.Tensor] = None,
        template_topk_identity: Optional[torch.Tensor] = None,
        template_topk_similarity: Optional[torch.Tensor] = None,
        template_topk_residue_idx: Optional[torch.Tensor] = None,
        template_chunk_coords: Optional[torch.Tensor] = None,
        template_chunk_mask: Optional[torch.Tensor] = None,
        template_chunk_start: Optional[torch.Tensor] = None,
        template_chunk_window_valid: Optional[torch.Tensor] = None,
        template_chunk_valid: Optional[torch.Tensor] = None,
        template_chunk_identity: Optional[torch.Tensor] = None,
        template_chunk_similarity: Optional[torch.Tensor] = None,
        template_chunk_confidence: Optional[torch.Tensor] = None,
        template_chunk_source_onehot: Optional[torch.Tensor] = None,
        template_chunk_residue_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del template_coords, template_mask, template_topk_coords, template_topk_mask
        del template_topk_valid, template_topk_identity, template_topk_similarity, template_topk_residue_idx

        b, l, _ = coords.shape
        if l > self.max_sequence_positions:
            raise ValueError(
                f"Sequence length {l} exceeds architectural limit {self.max_sequence_positions} "
                f"(chunk_length={self.template_chunk_length}, stride={self.template_chunk_stride}, max_windows={self.template_chunk_max_windows})."
            )

        rid = residue_idx.clamp(min=0, max=self.residue_emb.num_embeddings - 1)
        cid = chain_idx.clamp(min=0, max=self.chain_emb.num_embeddings - 1)
        pid = copy_idx.clamp(min=0, max=self.copy_emb.num_embeddings - 1)
        seq_embed = (
            self.residue_emb(rid)
            + self.chain_emb(cid)
            + self.copy_emb(pid)
            + self.resid_proj(resid.unsqueeze(-1).to(dtype=coords.dtype))
        ) * mask.unsqueeze(-1).float()

        (
            c,
            chunk_mask,
            chunk_start,
            chunk_window_valid,
            chunk_valid,
            chunk_identity,
            chunk_similarity,
            chunk_conf,
            chunk_source,
            chunk_residue,
        ) = self._prepare_chunk_inputs(
            coords=coords,
            mask=mask,
            residue_idx=rid,
            template_chunk_coords=template_chunk_coords,
            template_chunk_mask=template_chunk_mask,
            template_chunk_start=template_chunk_start,
            template_chunk_window_valid=template_chunk_window_valid,
            template_chunk_valid=template_chunk_valid,
            template_chunk_identity=template_chunk_identity,
            template_chunk_similarity=template_chunk_similarity,
            template_chunk_confidence=template_chunk_confidence,
            template_chunk_source_onehot=template_chunk_source_onehot,
            template_chunk_residue_idx=template_chunk_residue_idx,
        )

        b, w, k, clen, _ = c.shape
        pos_ids = torch.arange(clen, device=coords.device, dtype=torch.long).clamp(max=self.template_chunk_length - 1)
        pos_emb = self.chunk_pos_emb(pos_ids).view(1, 1, 1, clen, self.hidden_dim)
        w_ids = torch.arange(w, device=coords.device, dtype=torch.long).clamp(max=self.template_chunk_max_windows - 1)
        w_emb = self.window_emb(w_ids).view(1, w, 1, 1, self.hidden_dim)

        abs_idx = chunk_start.unsqueeze(-1) + torch.arange(clen, device=coords.device, dtype=torch.long).view(1, 1, clen)
        if l > 0:
            abs_idx_clamped = abs_idx.clamp(min=0, max=l - 1)
            target_residue = torch.gather(rid, 1, abs_idx_clamped.view(b, -1)).view(b, w, clen)
            target_resid = torch.gather(resid.to(dtype=coords.dtype), 1, abs_idx_clamped.view(b, -1)).view(b, w, clen)
            target_xyz = (
                torch.gather(
                    coords,
                    1,
                    abs_idx_clamped.view(b, -1).unsqueeze(-1).expand(-1, -1, 3),
                )
                .view(b, w, clen, 3)
                .to(dtype=coords.dtype)
            )
            target_mask = torch.gather(mask.long(), 1, abs_idx_clamped.view(b, -1)).view(b, w, clen).bool()
        else:
            abs_idx_clamped = abs_idx
            target_residue = torch.zeros(b, w, clen, device=coords.device, dtype=torch.long)
            target_resid = torch.zeros(b, w, clen, device=coords.device, dtype=coords.dtype)
            target_xyz = torch.zeros(b, w, clen, 3, device=coords.device, dtype=coords.dtype)
            target_mask = torch.zeros(b, w, clen, device=coords.device, dtype=torch.bool)

        valid_token = chunk_mask & chunk_window_valid.unsqueeze(-1) & (abs_idx < l) & target_mask

        target_token = (
            self.residue_emb(target_residue.clamp(min=0, max=self.residue_emb.num_embeddings - 1))
            + self.resid_proj(target_resid.unsqueeze(-1))
            + self.chunk_pos_emb(pos_ids).view(1, 1, clen, self.hidden_dim)
        )

        candidate_residue = self.chunk_residue_emb(
            chunk_residue.clamp(min=0, max=self.chunk_residue_emb.num_embeddings - 1)
        )
        coverage = chunk_mask.float().mean(dim=-1, keepdim=True).expand(-1, -1, k)
        meta = torch.stack(
            [
                (chunk_identity / 100.0).clamp(min=0.0, max=2.0),
                chunk_similarity.clamp(min=0.0, max=2.0),
                chunk_conf.clamp(min=0.0, max=1.5),
                chunk_source[..., 0].clamp(min=0.0, max=1.0),
                chunk_source[..., 1].clamp(min=0.0, max=1.0),
                chunk_valid.float(),
                coverage,
            ],
            dim=-1,
        )

        candidate_token = (
            self.chunk_coord_proj(c)
            + candidate_residue
            + target_token.unsqueeze(2)
            + pos_emb
            + w_emb
            + self.match_meta_proj(meta).unsqueeze(-2)
        ) * (chunk_valid.unsqueeze(-1).unsqueeze(-1).float())

        # Local graph encoder per candidate chunk.
        flat_h = candidate_token.reshape(b * w * k, clen, self.hidden_dim)
        flat_xyz = c.reshape(b * w * k, clen, 3)
        flat_valid = (chunk_valid.unsqueeze(-1) & valid_token.unsqueeze(2)).reshape(b * w * k, clen)

        empty_rows = ~flat_valid.any(dim=1)
        if bool(empty_rows.any().item()):
            flat_h = flat_h.clone()
            flat_xyz = flat_xyz.clone()
            flat_valid = flat_valid.clone()
            flat_h[empty_rows, 0] = 0.0
            flat_xyz[empty_rows, 0] = 0.0
            flat_valid[empty_rows, 0] = True

        for layer in self.graph_encoder:
            flat_h = layer(flat_h, flat_xyz, flat_valid)

        candidate_token = flat_h.view(b, w, k, clen, self.hidden_dim)

        candidate_coords_c = c.permute(0, 1, 3, 2, 4)          # (B, W, C, K, 3)
        cand_token_valid = chunk_valid.unsqueeze(-1) & valid_token.unsqueeze(2)  # (B, W, K, C)
        cand_token_valid_c = cand_token_valid.permute(0, 1, 3, 2)                # (B, W, C, K)
        candidate_token_c = candidate_token.permute(0, 1, 3, 2, 4)               # (B, W, C, K, H)

        role_ids = torch.arange(k + 1, device=coords.device, dtype=torch.long).clamp(
            max=self.score_role_emb.num_embeddings - 1
        )

        score_tokens = torch.cat([target_token.unsqueeze(3), candidate_token_c], dim=3)  # (B, W, C, K+1, H)
        score_mask = torch.cat([valid_token.unsqueeze(-1), cand_token_valid_c], dim=-1)   # (B, W, C, K+1)

        # Global sequence context.
        global_sum = torch.zeros(b, l, self.hidden_dim, device=coords.device, dtype=score_tokens.dtype)
        global_count = torch.zeros(b, l, device=coords.device, dtype=score_tokens.dtype)
        if l > 0:
            for role_idx in range(k + 1):
                role_hidden = score_tokens[:, :, :, role_idx, :].reshape(b, -1, self.hidden_dim)
                role_valid = score_mask[:, :, :, role_idx].reshape(b, -1)
                abs_flat = abs_idx_clamped.reshape(b, -1)
                for bi in range(b):
                    if not bool(role_valid[bi].any().item()):
                        continue
                    idx_sel = abs_flat[bi][role_valid[bi]]
                    hidden_sel = role_hidden[bi][role_valid[bi]]
                    global_sum[bi].index_add_(0, idx_sel, hidden_sel)
                    global_count[bi].index_add_(
                        0,
                        idx_sel,
                        torch.ones(idx_sel.shape[0], device=coords.device, dtype=score_tokens.dtype),
                    )

        global_context = torch.where(
            global_count.unsqueeze(-1) > 0.0,
            global_sum / global_count.unsqueeze(-1).clamp(min=1e-6),
            seq_embed,
        )
        global_context = self.score_global_norm(global_context + seq_embed)

        safe_global = global_context
        safe_global_mask = mask
        empty_global = ~safe_global_mask.any(dim=1)
        if bool(empty_global.any().item()):
            safe_global = safe_global.clone()
            safe_global_mask = safe_global_mask.clone()
            safe_global[empty_global, 0] = 0.0
            safe_global_mask[empty_global, 0] = True

        global_context = self.score_global_transformer(safe_global, src_key_padding_mask=~safe_global_mask)
        global_context = global_context * mask.unsqueeze(-1).float()

        if l > 0:
            token_global = (
                torch.gather(
                    global_context,
                    1,
                    abs_idx_clamped.view(b, -1).unsqueeze(-1).expand(-1, -1, self.hidden_dim),
                )
                .view(b, w, clen, self.hidden_dim)
                .to(dtype=score_tokens.dtype)
            )
        else:
            token_global = torch.zeros(b, w, clen, self.hidden_dim, device=coords.device, dtype=score_tokens.dtype)

        score_tokens = self.score_norm(
            score_tokens
            + token_global.unsqueeze(3)
            + self.score_role_emb(role_ids).view(1, 1, 1, k + 1, self.hidden_dim)
        )

        # Token-level scoring.
        flat_tokens = score_tokens.reshape(b * w * clen, k + 1, self.hidden_dim)
        flat_mask = score_mask.reshape(b * w * clen, k + 1)
        empty_rows = ~flat_mask.any(dim=1)
        if bool(empty_rows.any().item()):
            flat_tokens = flat_tokens.clone()
            flat_mask = flat_mask.clone()
            flat_tokens[empty_rows, 0] = 0.0
            flat_mask[empty_rows, 0] = True
        encoded_tokens = self.score_transformer(flat_tokens, src_key_padding_mask=~flat_mask)
        encoded_tokens = encoded_tokens.view(b, w, clen, k + 1, self.hidden_dim)

        token_logits = self.token_score_head(encoded_tokens[:, :, :, 1:, :]).squeeze(-1)
        token_scores = self._normalize_candidate_probs(token_logits, cand_token_valid_c)

        # Window-level scoring.
        encoded_ref = encoded_tokens[:, :, :, 0, :]
        encoded_cand = encoded_tokens[:, :, :, 1:, :].permute(0, 1, 3, 2, 4)
        token_weight = valid_token.float()
        token_count = token_weight.sum(dim=2, keepdim=True).clamp(min=1.0)
        ref_summary = (encoded_ref * token_weight.unsqueeze(-1)).sum(dim=2) / token_count
        cand_summary = (
            encoded_cand * token_weight.unsqueeze(2).unsqueeze(-1)
        ).sum(dim=3) / token_count.unsqueeze(2)
        global_summary = (token_global * token_weight.unsqueeze(-1)).sum(dim=2) / token_count

        window_tokens = torch.cat([ref_summary.unsqueeze(2), cand_summary], dim=2) + global_summary.unsqueeze(2)
        window_tokens = self.score_norm(window_tokens + self.score_role_emb(role_ids).view(1, 1, k + 1, self.hidden_dim))
        window_valid = chunk_valid & chunk_window_valid.unsqueeze(-1)
        window_token_mask = torch.cat([chunk_window_valid.unsqueeze(-1), window_valid], dim=2)

        flat_w_tokens = window_tokens.reshape(b * w, k + 1, self.hidden_dim)
        flat_w_mask = window_token_mask.reshape(b * w, k + 1)
        empty_w = ~flat_w_mask.any(dim=1)
        if bool(empty_w.any().item()):
            flat_w_tokens = flat_w_tokens.clone()
            flat_w_mask = flat_w_mask.clone()
            flat_w_tokens[empty_w, 0] = 0.0
            flat_w_mask[empty_w, 0] = True
        encoded_windows = self.score_transformer(flat_w_tokens, src_key_padding_mask=~flat_w_mask)
        encoded_windows = encoded_windows.view(b, w, k + 1, self.hidden_dim)

        window_logits = self.window_score_head(encoded_windows[:, :, 1:, :]).squeeze(-1)
        window_scores = self._normalize_candidate_probs(window_logits, window_valid)

        # Combined sparse mixture.
        combined_scores = token_scores * window_scores.unsqueeze(2)
        combined_scores = self._normalize_candidate_probs(
            logits=torch.log(combined_scores.clamp(min=1e-8)),
            valid=cand_token_valid_c,
        )
        combined_scores = self._sparsify_candidate_probs(combined_scores, cand_token_valid_c)

        window_pred = (combined_scores.unsqueeze(-1) * candidate_coords_c).sum(dim=3)  # (B, W, C, 3)
        token_conf = combined_scores.max(dim=-1).values  # (B, W, C)

        # Seam-aware weighted stitching.
        seam_base = F.softplus(self.seam_logits[:clen]) + 1e-3
        if clen > 1:
            axis = torch.linspace(-1.0, 1.0, steps=clen, device=coords.device, dtype=coords.dtype)
            center = 1.0 - axis.abs()
            center = 0.5 + 0.5 * center
        else:
            center = torch.ones((1,), device=coords.device, dtype=coords.dtype)
        seam_weights = seam_base * center

        aligned_window_pred = window_pred.clone()
        stitched_sum = torch.zeros(b, l, 3, device=coords.device, dtype=coords.dtype)
        stitched_w = torch.zeros(b, l, device=coords.device, dtype=coords.dtype)
        stitched_conf_sum = torch.zeros(b, l, device=coords.device, dtype=coords.dtype)

        for bi in range(b):
            active_windows = torch.nonzero(chunk_window_valid[bi], as_tuple=False).squeeze(-1)
            if int(active_windows.numel()) > 1:
                active_windows = active_windows[torch.argsort(chunk_start[bi, active_windows])]
            for wi_t in active_windows:
                wi = int(wi_t.item())
                if not bool(chunk_window_valid[bi, wi].item()):
                    continue
                token_idx = torch.nonzero(valid_token[bi, wi], as_tuple=False).squeeze(-1)
                if int(token_idx.numel()) <= 0:
                    continue
                abs_pos = abs_idx[bi, wi, token_idx]
                in_bounds = (abs_pos >= 0) & (abs_pos < l)
                if not bool(in_bounds.any().item()):
                    continue
                token_idx = token_idx[in_bounds]
                abs_pos = abs_pos[in_bounds]
                seg_coords = window_pred[bi, wi, token_idx]
                seg_conf = token_conf[bi, wi, token_idx]
                seam_score = seam_weights[token_idx] * (0.5 + 0.5 * seg_conf).clamp(min=1e-3)

                overlap_mask = stitched_w[bi, abs_pos] > 0.0
                if self.rigid_alignment_enabled and int(overlap_mask.sum().item()) >= 3:
                    src_ov = seg_coords[overlap_mask]
                    ref_ov = stitched_sum[bi, abs_pos[overlap_mask]] / stitched_w[bi, abs_pos[overlap_mask]].unsqueeze(
                        -1
                    ).clamp(min=1e-6)
                    ov_w = seam_score[overlap_mask] * stitched_w[bi, abs_pos[overlap_mask]].clamp(min=1e-6)
                    r, t = self._weighted_kabsch_transform(source=src_ov, target=ref_ov, weights=ov_w)
                    seg_coords = seg_coords @ r.transpose(0, 1) + t

                aligned_window_pred[bi, wi, token_idx] = seg_coords
                stitched_sum[bi].index_add_(0, abs_pos, seg_coords * seam_score.unsqueeze(-1))
                stitched_w[bi].index_add_(0, abs_pos, seam_score)
                stitched_conf_sum[bi].index_add_(0, abs_pos, seg_conf * seam_score)

        stitched_coords = torch.where(
            stitched_w.unsqueeze(-1) > 0.0,
            stitched_sum / stitched_w.unsqueeze(-1).clamp(min=1e-6),
            coords,
        )
        stitched_conf = torch.where(
            stitched_w > 0.0,
            stitched_conf_sum / stitched_w.clamp(min=1e-6),
            torch.zeros_like(stitched_conf_sum),
        )
        stitched_coords = stitched_coords * mask.unsqueeze(-1).float()
        stitched_conf = stitched_conf * mask.float()

        if w > 1 and clen > self.template_chunk_stride:
            ov = clen - self.template_chunk_stride
            left_ok = valid_token[:, :-1, clen - ov : clen]
            right_ok = valid_token[:, 1:, :ov]
            ov_valid = left_ok & right_ok
            left_xyz = aligned_window_pred[:, :-1, clen - ov : clen]
            right_xyz = aligned_window_pred[:, 1:, :ov]
            ov_gap = torch.linalg.norm(left_xyz - right_xyz, dim=-1)
            compat = (torch.exp(-0.25 * ov_gap) * ov_valid.float()).sum(dim=-1) / ov_valid.float().sum(dim=-1).clamp(
                min=1.0
            )
            self.last_overlap_compatibility = compat.detach()
        else:
            self.last_overlap_compatibility = torch.zeros(
                b,
                max(0, w - 1),
                device=coords.device,
                dtype=coords.dtype,
            )

        hidden = seq_embed + self.coord_context_proj(stitched_coords) + self.conf_context_proj(
            stitched_conf.unsqueeze(-1)
        )
        safe_hidden = hidden
        safe_mask = mask
        empty_seq = ~safe_mask.any(dim=1)
        if bool(empty_seq.any().item()):
            safe_hidden = safe_hidden.clone()
            safe_mask = safe_mask.clone()
            safe_hidden[empty_seq, 0] = 0.0
            safe_mask[empty_seq, 0] = True

        h = safe_hidden
        if self.refinement_block == "transformer":
            for layer in self.global_layers:
                h = layer(h, src_key_padding_mask=~safe_mask)
                h = h * safe_mask.unsqueeze(-1).float()
            assert self.coord_delta_head is not None
            pred = (stitched_coords + self.coord_delta_head(self.refine_norm(h))) * mask.unsqueeze(-1).float()
        else:
            pred = stitched_coords
            for layer in self.se3_layers:
                h, delta = layer(h, pred, mask)
                pred = (pred + delta) * mask.unsqueeze(-1).float()
                assert self.coord_to_hidden is not None
                h = (h + self.coord_to_hidden(delta)) * mask.unsqueeze(-1).float()

        model_conf = torch.sigmoid(self.conf_out(self.refine_norm(h))).squeeze(-1)
        self.last_confidence = (0.5 * model_conf + 0.5 * stitched_conf) * mask.float()
        self.last_token_candidate_scores = combined_scores.detach()
        self.last_window_candidate_scores = window_scores.detach()
        self.last_aux_losses = self._compute_geom_aux(pred, mask) if self.use_geom_aux else {}
        return pred


if __name__ == "__main__":
    _ = TemplateSegmentAssembler()
