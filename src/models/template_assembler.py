from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
from torch import nn

from src.models.components.folding_transformer import SE3RefinementBlock


class EGNNRefineLayer(nn.Module):
    """Lightweight EGNN layer with sparse sequence+geometric neighborhoods."""

    def __init__(
        self,
        hidden_dim: int,
        seq_radius: int = 2,
        knn: int = 16,
        coord_step_size: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_radius = max(0, int(seq_radius))
        self.knn = max(0, int(knn))
        self.coord_step_size = float(coord_step_size)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.h_norm = nn.LayerNorm(hidden_dim)

    def _build_edges(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build directed edges with sequence local links and geometric kNN links."""
        n = int(coords.shape[0])
        if n <= 1:
            empty = torch.zeros(0, device=coords.device, dtype=torch.long)
            return empty, empty

        src_chunks: list[torch.Tensor] = []
        dst_chunks: list[torch.Tensor] = []
        idx = torch.arange(n, device=coords.device, dtype=torch.long)

        if self.seq_radius > 0:
            for off in range(1, self.seq_radius + 1):
                if off >= n:
                    break
                s_f = idx[:-off]
                d_f = idx[off:]
                src_chunks.extend([s_f, d_f])
                dst_chunks.extend([d_f, s_f])

        if self.knn > 0:
            k = min(self.knn, n - 1)
            dist = torch.cdist(coords.to(dtype=torch.float32), coords.to(dtype=torch.float32))
            dist.fill_diagonal_(float("inf"))
            nn_idx = torch.topk(dist, k=k, largest=False, dim=-1).indices  # (N, k)
            src_geo = idx.unsqueeze(1).expand(-1, k).reshape(-1)
            dst_geo = nn_idx.reshape(-1)
            src_chunks.append(src_geo)
            dst_chunks.append(dst_geo)

        if not src_chunks:
            empty = torch.zeros(0, device=coords.device, dtype=torch.long)
            return empty, empty

        src = torch.cat(src_chunks, dim=0)
        dst = torch.cat(dst_chunks, dim=0)
        linear = src * n + dst
        uniq = torch.unique(linear, sorted=True)
        src = uniq // n
        dst = uniq % n
        return src, dst

    def forward(self, hidden: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_h = hidden.clone()
        out_x = coords.clone()

        for b in range(hidden.shape[0]):
            valid_idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
            n = int(valid_idx.numel())
            if n <= 1:
                continue

            h_b = hidden[b, valid_idx, :]
            x_b = coords[b, valid_idx, :]
            src, dst = self._build_edges(x_b)
            if src.numel() == 0:
                continue

            h_src = h_b[src]
            h_dst = h_b[dst]
            rel = x_b[src] - x_b[dst]
            dist2 = torch.sum(rel * rel, dim=-1, keepdim=True)
            msg = self.edge_mlp(torch.cat([h_src, h_dst, dist2], dim=-1))

            coord_coef = self.coord_mlp(msg)
            delta = rel * coord_coef
            agg_delta = torch.zeros(n, 3, device=coords.device, dtype=coords.dtype)
            agg_delta.index_add_(0, src, delta.to(dtype=agg_delta.dtype))
            deg = torch.zeros(n, 1, device=coords.device, dtype=coords.dtype)
            deg.index_add_(0, src, torch.ones(src.shape[0], 1, device=coords.device, dtype=coords.dtype))
            x_new = x_b + self.coord_step_size * agg_delta / deg.clamp(min=1.0)

            agg_msg = torch.zeros(n, self.hidden_dim, device=hidden.device, dtype=hidden.dtype)
            agg_msg.index_add_(0, src, msg.to(dtype=hidden.dtype))
            h_new = h_b + self.node_mlp(torch.cat([h_b, agg_msg], dim=-1))
            h_new = self.h_norm(h_new)

            out_h[b, valid_idx, :] = h_new
            out_x[b, valid_idx, :] = x_new

        return out_h, out_x


class TemplateSegmentAssembler(nn.Module):
    """Local template picker + equivariant refinement.

    The model expects five candidate 3D structures per target (top identity matches).
    It computes soft candidate selection per segment, stitches a consensus backbone,
    then applies EGNN + a small SE(3) refinement pass.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_candidates: int = 5,
        segment_length: int = 64,
        segment_stride: int = 32,
        residue_vocab_size: int = 5,
        max_chain_embeddings: int = 64,
        max_copy_embeddings: int = 64,
        egnn_layers: int = 3,
        egnn_seq_radius: int = 2,
        egnn_knn: int = 16,
        egnn_coord_step_size: float = 0.1,
        se3_refine_layers: int = 1,
        se3_num_heads: int = 4,
        se3_dropout: float = 0.0,
        se3_coord_step_size: float = 0.1,
        template_chunk_length: int = 512,
        template_chunk_stride: int = 256,
        template_chunk_max_windows: int = 20,
        hybrid_gru_layers: int = 1,
        hybrid_gru_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_candidates = num_candidates
        self.segment_length = max(8, int(segment_length))
        self.segment_stride = max(1, int(segment_stride))
        self.template_chunk_length = max(1, int(template_chunk_length))
        self.template_chunk_stride = max(1, int(template_chunk_stride))
        self.template_chunk_max_windows = max(1, int(template_chunk_max_windows))
        self.invariant_feature_dim = 8

        self.residue_emb = nn.Embedding(residue_vocab_size, hidden_dim)
        self.chain_emb = nn.Embedding(max_chain_embeddings, hidden_dim)
        self.copy_emb = nn.Embedding(max_copy_embeddings, hidden_dim)
        self.resid_proj = nn.Linear(1, hidden_dim)
        self.coord_proj = nn.Linear(3, hidden_dim)
        self.quality_proj = nn.Linear(2, hidden_dim)
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.invariant_norm = nn.LayerNorm(self.invariant_feature_dim)
        self.invariant_gru = nn.GRU(
            input_size=self.invariant_feature_dim,
            hidden_size=hidden_dim,
            num_layers=max(1, int(hybrid_gru_layers)),
            dropout=float(hybrid_gru_dropout) if int(hybrid_gru_layers) > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.invariant_gru_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.invariant_gate = nn.Sequential(
            nn.Linear(hidden_dim + self.invariant_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.segment_scorer = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.egnn_layers = nn.ModuleList(
            [
                EGNNRefineLayer(
                    hidden_dim=hidden_dim,
                    seq_radius=egnn_seq_radius,
                    knn=egnn_knn,
                    coord_step_size=egnn_coord_step_size,
                )
                for _ in range(egnn_layers)
            ]
        )

        self.se3_layers = nn.ModuleList(
            [
                SE3RefinementBlock(
                    hidden_dim=hidden_dim,
                    num_heads=se3_num_heads,
                    dropout=se3_dropout,
                    coord_step_size=se3_coord_step_size,
                    use_fast_multipole=False,
                    use_block_sparse_attention=True,
                    block_sparse_min_seq_len=1,
                    block_sparse_block_size=64,
                    block_sparse_window=128,
                    block_sparse_global_stride=128,
                    block_sparse_max_global=32,
                    block_sparse_geo_topk=16,
                    pure_sparse_mode=True,
                )
                for _ in range(se3_refine_layers)
            ]
        )

    def _segment_spans(self, length: int) -> list[tuple[int, int]]:
        if length <= self.segment_length:
            return [(0, length)]
        spans: list[tuple[int, int]] = []
        start = 0
        while start < length:
            end = min(start + self.segment_length, length)
            spans.append((start, end))
            if end == length:
                break
            start += self.segment_stride
        last_start = max(0, length - self.segment_length)
        if spans[-1][0] != last_start:
            spans.append((last_start, length))
        return spans

    def _kabsch_align_to_reference(
        self,
        candidates: torch.Tensor,
        reference: torch.Tensor,
        mask: torch.Tensor,
        valid_cands: torch.Tensor,
    ) -> torch.Tensor:
        aligned = candidates.clone()
        use_fp32_linalg = candidates.device.type == "cuda" and (
            candidates.dtype in (torch.float16, torch.bfloat16) or torch.is_autocast_enabled()
        )

        autocast_ctx = (
            torch.autocast(device_type=candidates.device.type, enabled=False)
            if use_fp32_linalg and candidates.device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with torch.no_grad():
            with autocast_ctx:
                bsz, k, _, _ = candidates.shape
                for b in range(bsz):
                    valid_res = mask[b]
                    if int(valid_res.sum().item()) < 3:
                        continue
                    target = reference[b, valid_res].to(dtype=torch.float32)
                    target_center = target.mean(dim=0, keepdim=True)
                    target_centered = target - target_center

                    for c in range(k):
                        if not bool(valid_cands[b, c].item()):
                            continue
                        pred_sel = candidates[b, c, valid_res].to(dtype=torch.float32)
                        if pred_sel.shape[0] < 3:
                            continue
                        pred_center = pred_sel.mean(dim=0, keepdim=True)
                        pred_centered = pred_sel - pred_center

                        cov = pred_centered.transpose(0, 1) @ target_centered
                        u, _, vh = torch.linalg.svd(cov, full_matrices=False)
                        r = vh.transpose(0, 1) @ u.transpose(0, 1)
                        if torch.det(r) < 0:
                            vh_fix = vh.clone()
                            vh_fix[-1, :] *= -1
                            r = vh_fix.transpose(0, 1) @ u.transpose(0, 1)

                        pred_all = candidates[b, c].to(dtype=torch.float32)
                        pred_all_aligned = (pred_all - pred_center) @ r + target_center
                        aligned[b, c] = pred_all_aligned.to(dtype=aligned.dtype)
        return aligned

    def _prepare_candidates(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        template_coords: Optional[torch.Tensor],
        template_mask: Optional[torch.Tensor],
        template_topk_coords: Optional[torch.Tensor],
        template_topk_mask: Optional[torch.Tensor],
        template_topk_valid: Optional[torch.Tensor],
        template_topk_identity: Optional[torch.Tensor],
        template_topk_similarity: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = coords.shape
        k = self.num_candidates

        candidates = coords.unsqueeze(1).repeat(1, k, 1, 1)
        valid = torch.zeros(bsz, k, device=coords.device, dtype=torch.bool)
        cand_identity = torch.zeros(bsz, k, device=coords.device, dtype=coords.dtype)
        cand_similarity = torch.zeros(bsz, k, device=coords.device, dtype=coords.dtype)
        valid[:, 0] = True
        cand_identity[:, 0] = 100.0
        cand_similarity[:, 0] = 1.0

        if template_topk_coords is not None:
            topk = template_topk_coords
            if topk.dim() != 4:
                raise ValueError("template_topk_coords must have shape (B,K,L,3) or (B,L,K,3)")
            if topk.shape[1] == seq_len and topk.shape[2] != seq_len:
                topk = topk.permute(0, 2, 1, 3)
            if topk.shape[2] != seq_len:
                raise ValueError("template_topk_coords sequence length does not match coords.")

            use_k = min(k, topk.shape[1])
            candidates[:, :use_k] = topk[:, :use_k]

            if template_topk_valid is not None and template_topk_valid.dim() == 2:
                valid[:, :use_k] = template_topk_valid[:, :use_k].to(dtype=torch.bool, device=coords.device)
            elif template_topk_mask is not None and template_topk_mask.dim() == 3:
                # (B, K, L) availability-by-token
                valid[:, :use_k] = template_topk_mask[:, :use_k].any(dim=-1).to(dtype=torch.bool, device=coords.device)
            else:
                valid[:, :use_k] = True

            if template_topk_identity is not None and template_topk_identity.dim() == 2:
                cand_identity[:, :use_k] = template_topk_identity[:, :use_k].to(dtype=coords.dtype, device=coords.device)
            if template_topk_similarity is not None and template_topk_similarity.dim() == 2:
                cand_similarity[:, :use_k] = template_topk_similarity[:, :use_k].to(
                    dtype=coords.dtype, device=coords.device
                )
        elif template_coords is not None:
            candidates[:, 1] = template_coords
            if template_mask is not None:
                valid[:, 1] = template_mask.any(dim=-1).to(dtype=torch.bool, device=coords.device)
            else:
                valid[:, 1] = True

        # Guarantee at least one valid candidate.
        empty_rows = ~valid.any(dim=1)
        if bool(empty_rows.any()):
            candidates[empty_rows, 0] = coords[empty_rows]
            valid[empty_rows, 0] = True
            cand_identity[empty_rows, 0] = 100.0
            cand_similarity[empty_rows, 0] = 1.0

        candidates = self._kabsch_align_to_reference(candidates, coords, mask, valid)
        return candidates, valid, cand_identity, cand_similarity

    def _segment_logits(
        self,
        candidates: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        valid: torch.Tensor,
        cand_identity: torch.Tensor,
        cand_similarity: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        bsz, k, seq_len, _ = candidates.shape
        spans = self._segment_spans(seq_len)
        logits_all = []
        valid_f = valid.float()
        valid_norm = valid_f.sum(dim=1, keepdim=True).clamp(min=1.0)

        for start, end in spans:
            seg_mask = mask[:, start:end].float()  # (B, S)
            seg_len = seg_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
            seg_coords = candidates[:, :, start:end, :]  # (B,K,S,3)
            seg_mean = (seg_coords * seg_mask[:, None, :, None]).sum(dim=2) / seg_len.unsqueeze(-1)

            input_seg = coords[:, start:end, :]
            input_seg_mean = (input_seg * seg_mask[:, :, None]).sum(dim=1) / seg_len

            consensus_seg = (seg_mean * valid_f[:, :, None]).sum(dim=1) / valid_norm
            dev_consensus = torch.linalg.norm(seg_mean - consensus_seg[:, None, :], dim=-1)
            dev_input = torch.linalg.norm(seg_mean - input_seg_mean[:, None, :], dim=-1)
            sq_dist = ((seg_coords - seg_mean[:, :, None, :]) ** 2).sum(dim=-1)  # (B,K,S)
            seg_spread = torch.sqrt(
                (sq_dist * seg_mask[:, None, :]).sum(dim=-1) / seg_len + 1e-6
            )
            id_feat = (cand_identity / 100.0).clamp(min=0.0, max=1.5)
            sim_feat = cand_similarity.clamp(min=0.0, max=2.0)

            feat = torch.stack(
                [
                    -dev_consensus,
                    -dev_input,
                    -seg_spread,
                    valid_f,
                    id_feat,
                    sim_feat,
                ],
                dim=-1,
            )  # (B,K,6)
            seg_logits = self.segment_scorer(feat).squeeze(-1)  # (B,K)
            seg_logits = seg_logits.masked_fill(~valid, -1e4)
            logits_all.append(seg_logits)

        return torch.stack(logits_all, dim=1), spans  # (B,M,K)

    def _stitch_consensus(
        self,
        candidates: torch.Tensor,
        seg_logits: torch.Tensor,
        spans: list[tuple[int, int]],
        mask: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, k, seq_len, _ = candidates.shape
        logits_acc = torch.zeros(bsz, seq_len, k, device=candidates.device, dtype=candidates.dtype)
        counts = torch.zeros(bsz, seq_len, 1, device=candidates.device, dtype=candidates.dtype)

        for seg_idx, (start, end) in enumerate(spans):
            logits_acc[:, start:end, :] += seg_logits[:, seg_idx, :].unsqueeze(1)
            counts[:, start:end, :] += 1.0

        res_logits = logits_acc / counts.clamp(min=1.0)
        res_logits = res_logits.masked_fill(~valid.unsqueeze(1), -1e4)
        weights = torch.softmax(res_logits, dim=-1)  # (B,L,K)
        weights = weights * mask.unsqueeze(-1).float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        cand_l = candidates.permute(0, 2, 1, 3)  # (B,L,K,3)
        consensus = torch.sum(weights.unsqueeze(-1) * cand_l, dim=2)  # (B,L,3)
        return consensus, weights

    def _build_global_invariants(
        self,
        candidates: torch.Tensor,
        weights: torch.Tensor,
        cand_identity: torch.Tensor,
        cand_similarity: torch.Tensor,
        consensus: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        cand_l = candidates.permute(0, 2, 1, 3)  # (B,L,K,3)
        d2_consensus = ((cand_l - consensus.unsqueeze(2)) ** 2).sum(dim=-1)  # (B,L,K)
        spread = torch.sqrt((weights * d2_consensus).sum(dim=-1).clamp(min=1e-6))  # (B,L)
        displacement = torch.linalg.norm(consensus - coords, dim=-1)  # (B,L)
        id_feat = (weights * (cand_identity / 100.0).unsqueeze(1)).sum(dim=-1)  # (B,L)
        sim_feat = (weights * cand_similarity.unsqueeze(1)).sum(dim=-1)  # (B,L)
        confidence = weights.max(dim=-1).values  # (B,L)
        entropy = -(weights.clamp(min=1e-8) * weights.clamp(min=1e-8).log()).sum(dim=-1)  # (B,L)
        coverage = mask.float()  # Global path covers all valid tokens.

        inv = torch.stack(
            [
                confidence,
                entropy,
                id_feat.clamp(min=0.0, max=1.5),
                sim_feat.clamp(min=0.0, max=2.0),
                displacement,
                spread,
                coverage,
                mask.float(),
            ],
            dim=-1,
        )  # (B,L,8)
        return inv

    def _apply_invariant_gru(
        self, hidden: torch.Tensor, invariant_features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        inv = self.invariant_norm(invariant_features.to(dtype=hidden.dtype))
        lengths = mask.sum(dim=-1).to(dtype=torch.long)

        if int(lengths.min().item()) <= 0:
            gru_out, _ = self.invariant_gru(inv)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                inv,
                lengths=lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.invariant_gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=hidden.shape[1],
            )

        gru_hidden = self.invariant_gru_proj(gru_out)
        gate = self.invariant_gate(torch.cat([hidden, inv], dim=-1))
        fused = hidden + gate * gru_hidden
        fused = fused * mask.unsqueeze(-1).float()
        return fused

    def _stitch_chunk_consensus(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        template_chunk_coords: torch.Tensor,
        template_chunk_mask: Optional[torch.Tensor],
        template_chunk_start: Optional[torch.Tensor],
        template_chunk_window_valid: Optional[torch.Tensor],
        template_chunk_valid: Optional[torch.Tensor],
        template_chunk_identity: Optional[torch.Tensor],
        template_chunk_similarity: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = coords.shape
        _, max_windows, k, chunk_len, _ = template_chunk_coords.shape

        coords_sum = torch.zeros_like(coords)
        counts = torch.zeros(bsz, seq_len, 1, device=coords.device, dtype=coords.dtype)
        conf_sum = torch.zeros_like(counts)
        ent_sum = torch.zeros_like(counts)
        quality_counts = torch.zeros_like(counts)
        id_sum = torch.zeros_like(counts)
        sim_sum = torch.zeros_like(counts)
        disp_sum = torch.zeros_like(counts)
        spread_sum = torch.zeros_like(counts)

        full_true_mask = torch.ones(1, chunk_len, device=coords.device, dtype=torch.bool)

        for b in range(bsz):
            valid_len = int(mask[b].sum().item())
            if valid_len <= 0:
                continue

            for w in range(max_windows):
                if template_chunk_window_valid is not None and not bool(template_chunk_window_valid[b, w].item()):
                    continue

                if template_chunk_start is not None:
                    start = int(template_chunk_start[b, w].item())
                else:
                    start = min(w * self.template_chunk_stride, max(0, valid_len - self.template_chunk_length))
                if start >= valid_len:
                    continue

                if template_chunk_mask is not None:
                    win_len = int(template_chunk_mask[b, w].sum().item())
                else:
                    win_len = min(chunk_len, valid_len - start)
                win_len = max(0, min(win_len, valid_len - start, chunk_len))
                if win_len <= 0:
                    continue
                end = start + win_len

                cand_coords = template_chunk_coords[b, w, :, :win_len]  # (K,S,3)
                if template_chunk_valid is not None:
                    cand_valid = template_chunk_valid[b, w].to(dtype=torch.bool, device=coords.device)
                else:
                    cand_valid = torch.ones(k, dtype=torch.bool, device=coords.device)

                cand_identity = torch.zeros(k, device=coords.device, dtype=coords.dtype)
                cand_similarity = torch.zeros(k, device=coords.device, dtype=coords.dtype)
                if template_chunk_identity is not None:
                    cand_identity = template_chunk_identity[b, w].to(dtype=coords.dtype, device=coords.device)
                if template_chunk_similarity is not None:
                    cand_similarity = template_chunk_similarity[b, w].to(dtype=coords.dtype, device=coords.device)

                if not bool(cand_valid.any().item()):
                    # Fallback to no-change chunk if all template candidates are missing.
                    cand_valid = cand_valid.clone()
                    cand_valid[0] = True
                    cand_coords = cand_coords.clone()
                    cand_coords[0] = coords[b, start:end]
                    cand_identity[0] = 100.0
                    cand_similarity[0] = 1.0

                chunk_input = coords[b, start:end]
                aligned = self._kabsch_align_to_reference(
                    candidates=cand_coords.unsqueeze(0),
                    reference=chunk_input.unsqueeze(0),
                    mask=full_true_mask[:, :win_len],
                    valid_cands=cand_valid.unsqueeze(0),
                )[0]

                cand_mean = aligned.mean(dim=1)  # (K,3)
                input_mean = chunk_input.mean(dim=0, keepdim=False)  # (3,)
                valid_f = cand_valid.float()
                valid_norm = valid_f.sum().clamp(min=1.0)
                consensus_mean = (cand_mean * valid_f[:, None]).sum(dim=0) / valid_norm
                dev_consensus = torch.linalg.norm(cand_mean - consensus_mean[None, :], dim=-1)
                dev_input = torch.linalg.norm(cand_mean - input_mean[None, :], dim=-1)
                spread = torch.sqrt(((aligned - cand_mean[:, None, :]) ** 2).sum(dim=-1).mean(dim=-1) + 1e-6)
                id_feat = (cand_identity / 100.0).clamp(min=0.0, max=1.5)
                sim_feat = cand_similarity.clamp(min=0.0, max=2.0)

                feat = torch.stack(
                    [
                        -dev_consensus,
                        -dev_input,
                        -spread,
                        valid_f,
                        id_feat,
                        sim_feat,
                    ],
                    dim=-1,
                )  # (K,6)
                logits = self.segment_scorer(feat).squeeze(-1)
                logits = logits.masked_fill(~cand_valid, -1e4)
                weights = torch.softmax(logits, dim=0)  # (K,)

                fused = torch.sum(weights[:, None, None] * aligned, dim=0)  # (S,3)
                coords_sum[b, start:end] += fused
                counts[b, start:end] += 1.0

                conf_val = weights.max()
                ent_val = -(weights.clamp(min=1e-8) * weights.clamp(min=1e-8).log()).sum()
                conf_sum[b, start:end] += conf_val
                ent_sum[b, start:end] += ent_val
                quality_counts[b, start:end] += 1.0

                weighted_id = torch.sum(weights * (cand_identity / 100.0).clamp(min=0.0, max=1.5))
                weighted_sim = torch.sum(weights * cand_similarity.clamp(min=0.0, max=2.0))
                weighted_spread = torch.sum(weights * spread)
                disp_tok = torch.linalg.norm(fused - chunk_input, dim=-1, keepdim=True)
                id_sum[b, start:end] += weighted_id
                sim_sum[b, start:end] += weighted_sim
                spread_sum[b, start:end] += weighted_spread
                disp_sum[b, start:end] += disp_tok

        consensus = torch.where(counts > 0, coords_sum / counts.clamp(min=1e-6), coords)
        confidence = torch.where(
            quality_counts > 0, conf_sum / quality_counts.clamp(min=1e-6), torch.ones_like(conf_sum)
        )
        entropy = torch.where(quality_counts > 0, ent_sum / quality_counts.clamp(min=1e-6), torch.zeros_like(ent_sum))
        id_feat = torch.where(quality_counts > 0, id_sum / quality_counts.clamp(min=1e-6), torch.ones_like(id_sum))
        sim_feat = torch.where(quality_counts > 0, sim_sum / quality_counts.clamp(min=1e-6), torch.zeros_like(sim_sum))
        spread_feat = torch.where(
            quality_counts > 0, spread_sum / quality_counts.clamp(min=1e-6), torch.zeros_like(spread_sum)
        )
        disp_feat = torch.where(quality_counts > 0, disp_sum / quality_counts.clamp(min=1e-6), torch.zeros_like(disp_sum))

        coverage = torch.zeros_like(counts)
        max_cover = counts.amax(dim=1, keepdim=True).clamp(min=1.0)
        coverage = counts / max_cover

        quality = torch.cat([confidence, entropy], dim=-1)
        invariant_features = torch.cat(
            [
                confidence,
                entropy,
                id_feat.clamp(min=0.0, max=1.5),
                sim_feat.clamp(min=0.0, max=2.0),
                disp_feat,
                spread_feat,
                coverage,
                mask.unsqueeze(-1).float(),
            ],
            dim=-1,
        )  # (B,L,8)
        return consensus, quality, invariant_features

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
        template_chunk_coords: Optional[torch.Tensor] = None,
        template_chunk_mask: Optional[torch.Tensor] = None,
        template_chunk_start: Optional[torch.Tensor] = None,
        template_chunk_window_valid: Optional[torch.Tensor] = None,
        template_chunk_valid: Optional[torch.Tensor] = None,
        template_chunk_identity: Optional[torch.Tensor] = None,
        template_chunk_similarity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residue_idx = residue_idx.clamp_(min=0, max=self.residue_emb.num_embeddings - 1)
        chain_idx = chain_idx.clamp_(min=0, max=self.chain_emb.num_embeddings - 1)
        copy_idx = copy_idx.clamp_(min=0, max=self.copy_emb.num_embeddings - 1)

        if template_chunk_coords is not None and template_chunk_coords.dim() == 5:
            consensus, quality, invariant_features = self._stitch_chunk_consensus(
                coords=coords,
                mask=mask,
                template_chunk_coords=template_chunk_coords,
                template_chunk_mask=template_chunk_mask,
                template_chunk_start=template_chunk_start,
                template_chunk_window_valid=template_chunk_window_valid,
                template_chunk_valid=template_chunk_valid,
                template_chunk_identity=template_chunk_identity,
                template_chunk_similarity=template_chunk_similarity,
            )
        else:
            candidates, valid, cand_identity, cand_similarity = self._prepare_candidates(
                coords=coords,
                mask=mask,
                template_coords=template_coords,
                template_mask=template_mask,
                template_topk_coords=template_topk_coords,
                template_topk_mask=template_topk_mask,
                template_topk_valid=template_topk_valid,
                template_topk_identity=template_topk_identity,
                template_topk_similarity=template_topk_similarity,
            )

            seg_logits, spans = self._segment_logits(
                candidates=candidates,
                coords=coords,
                mask=mask,
                valid=valid,
                cand_identity=cand_identity,
                cand_similarity=cand_similarity,
            )
            consensus, cand_weights = self._stitch_consensus(
                candidates=candidates,
                seg_logits=seg_logits,
                spans=spans,
                mask=mask,
                valid=valid,
            )

            confidence = cand_weights.max(dim=-1).values.unsqueeze(-1)
            entropy = -(cand_weights.clamp(min=1e-8) * cand_weights.clamp(min=1e-8).log()).sum(dim=-1, keepdim=True)
            quality = torch.cat([confidence, entropy], dim=-1)
            invariant_features = self._build_global_invariants(
                candidates=candidates,
                weights=cand_weights,
                cand_identity=cand_identity,
                cand_similarity=cand_similarity,
                consensus=consensus,
                coords=coords,
                mask=mask,
            )

        hidden = (
            self.residue_emb(residue_idx)
            + self.chain_emb(chain_idx)
            + self.copy_emb(copy_idx)
            + self.resid_proj(resid.unsqueeze(-1))
            + self.coord_proj(consensus)
            + self.quality_proj(quality)
        )
        hidden = self.hidden_norm(hidden)
        hidden = hidden * mask.unsqueeze(-1).float()
        hidden = self._apply_invariant_gru(hidden, invariant_features, mask)

        refined = consensus
        for layer in self.egnn_layers:
            hidden, refined = layer(hidden, refined, mask)
            hidden = hidden * mask.unsqueeze(-1).float()
            refined = refined * mask.unsqueeze(-1).float()

        for layer in self.se3_layers:
            hidden, coord_update = layer(hidden, refined, mask)
            refined = refined + coord_update
            hidden = hidden + self.coord_proj(coord_update)
            hidden = hidden * mask.unsqueeze(-1).float()

        return refined


if __name__ == "__main__":
    _ = TemplateSegmentAssembler()
