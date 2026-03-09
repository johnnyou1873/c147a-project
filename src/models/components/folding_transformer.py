from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class SE3RefinementBlock(nn.Module):
    """Single equivariant refinement block.

    Coordinates are updated only via weighted sums of relative vectors `(x_j - x_i)`,
    which preserves SE(3) equivariance.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        coord_step_size: float = 0.25,
        use_fast_multipole: bool = True,
        fmm_cell_size: int = 64,
        fmm_near_cell_span: int = 1,
        fmm_exact_threshold: int = 768,
        fmm_max_levels: int = 4,
        fmm_far_topk: int = 32,
        fmm_dynamic_reordering: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")
        if fmm_cell_size < 1:
            raise ValueError("fmm_cell_size must be >= 1")
        if fmm_near_cell_span < 0:
            raise ValueError("fmm_near_cell_span must be >= 0")
        if fmm_max_levels < 1:
            raise ValueError("fmm_max_levels must be >= 1")
        if fmm_far_topk < 1:
            raise ValueError("fmm_far_topk must be >= 1")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.coord_step_size = coord_step_size
        self.use_fast_multipole = use_fast_multipole
        self.fmm_cell_size = fmm_cell_size
        self.fmm_near_cell_span = fmm_near_cell_span
        self.fmm_exact_threshold = fmm_exact_threshold
        self.fmm_max_levels = fmm_max_levels
        self.fmm_far_topk = fmm_far_topk
        self.fmm_dynamic_reordering = fmm_dynamic_reordering

        self.h_norm = nn.LayerNorm(hidden_dim)
        self.ff_norm = nn.LayerNorm(hidden_dim)

        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

        self.dist_bias = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.coord_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def _dist_bias(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: (B, Q, K) -> (B, H, Q, K) OR (Q, K) -> (H, Q, K)
        if dist.dim() == 3:
            pair_bias = self.dist_bias(dist.unsqueeze(-1))  # (B, Q, K, H)
            return pair_bias.permute(0, 3, 1, 2)
        if dist.dim() == 2:
            pair_bias = self.dist_bias(dist.unsqueeze(-1))  # (Q, K, H)
            return pair_bias.permute(2, 0, 1)
        raise ValueError(f"Unsupported distance tensor shape: {tuple(dist.shape)}")

    def _exact_message_and_delta(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rel = coords.unsqueeze(1) - coords.unsqueeze(2)  # (B, L, L, 3): x_j - x_i at [i,j]
        dist = torch.linalg.norm(rel, dim=-1).clamp(min=1e-6)  # (B, L, L)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_logits = attn_logits + self._dist_bias(dist)

        key_mask = mask[:, None, None, :]  # (B,1,1,L)
        query_mask = mask[:, None, :, None]  # (B,1,L,1)
        attn_logits = attn_logits.masked_fill(~key_mask, -1e4)
        attn = torch.softmax(attn_logits, dim=-1) * query_mask
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        msg = torch.matmul(attn, v).transpose(1, 2).reshape(mask.shape[0], mask.shape[1], self.hidden_dim)
        attn_mean = attn.mean(dim=1)  # (B, L, L)
        coord_delta = (attn_mean.unsqueeze(-1) * rel).sum(dim=2)  # (B, L, 3)
        return msg, coord_delta

    def _dynamic_reorder(self, coords: torch.Tensor) -> torch.Tensor:
        """Order residues along the principal spatial axis for sharper local groupings."""
        n = coords.shape[0]
        if n <= 1:
            return torch.arange(n, device=coords.device, dtype=torch.long)
        use_fp32_linalg = coords.device.type == "cuda" and (
            coords.dtype in (torch.float16, torch.bfloat16) or torch.is_autocast_enabled()
        )

        if use_fp32_linalg:
            # In fp16/16-mixed on CUDA, eigh for Half is unavailable.
            # Compute ordering in fp32 with autocast disabled.
            with torch.autocast(device_type=coords.device.type, enabled=False):
                coords_f = coords.to(dtype=torch.float32)
                centered = coords_f - coords_f.mean(dim=0, keepdim=True)
                cov = centered.transpose(0, 1) @ centered
                eigvals, eigvecs = torch.linalg.eigh(cov)
                principal_axis = eigvecs[:, torch.argmax(eigvals)]
                proj = centered @ principal_axis
            return torch.argsort(proj)

        centered = coords - coords.mean(dim=0, keepdim=True)
        cov = centered.transpose(0, 1) @ centered
        eigvals, eigvecs = torch.linalg.eigh(cov)
        principal_axis = eigvecs[:, torch.argmax(eigvals)]
        proj = centered @ principal_axis
        return torch.argsort(proj)

    def _build_hierarchy(
        self,
        k_b: torch.Tensor,
        v_b: torch.Tensor,
        x_b: torch.Tensor,
    ) -> list[dict[str, torch.Tensor | int]]:
        """Build multi-resolution cell summaries for hierarchical FMM."""
        levels: list[dict[str, torch.Tensor | int]] = []
        n = x_b.shape[0]
        cell_size = self.fmm_cell_size
        level = 0

        while level < self.fmm_max_levels:
            num_cells = (n + cell_size - 1) // cell_size
            if num_cells <= 0:
                break

            centers = torch.zeros(num_cells, 3, device=x_b.device, dtype=x_b.dtype)
            k_sum = torch.zeros(
                self.num_heads, num_cells, self.head_dim, device=k_b.device, dtype=k_b.dtype
            )
            v_sum = torch.zeros(
                self.num_heads, num_cells, self.head_dim, device=v_b.device, dtype=v_b.dtype
            )

            for c in range(num_cells):
                s = c * cell_size
                e = min((c + 1) * cell_size, n)
                centers[c] = x_b[s:e].mean(dim=0)
                k_sum[:, c, :] = k_b[:, s:e, :].mean(dim=1)
                v_sum[:, c, :] = v_b[:, s:e, :].mean(dim=1)

            levels.append(
                {
                    "cell_size": cell_size,
                    "num_cells": num_cells,
                    "centers": centers,
                    "k": k_sum,
                    "v": v_sum,
                }
            )
            if num_cells == 1:
                break
            cell_size *= 2
            level += 1

        return levels

    def _fmm_message_and_delta(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hierarchical FMM with dynamic reordering for sharper local interactions."""
        bsz, _, seq_len, _ = q.shape
        msg_out = torch.zeros(bsz, seq_len, self.hidden_dim, device=q.device, dtype=q.dtype)
        coord_delta_out = torch.zeros(bsz, seq_len, 3, device=coords.device, dtype=coords.dtype)
        sqrt_d = self.head_dim**0.5
        span = self.fmm_near_cell_span

        for b in range(bsz):
            valid_idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
            n = int(valid_idx.numel())
            if n == 0:
                continue

            x_valid = coords[b, valid_idx, :]
            if self.fmm_dynamic_reordering:
                reorder = self._dynamic_reorder(x_valid)
                ordered_idx = valid_idx[reorder]
            else:
                ordered_idx = valid_idx

            q_b = q[b, :, ordered_idx, :]  # (H, N, D)
            k_b = k[b, :, ordered_idx, :]
            v_b = v[b, :, ordered_idx, :]
            x_b = coords[b, ordered_idx, :]  # (N, 3)

            levels = self._build_hierarchy(k_b, v_b, x_b)
            if not levels:
                continue

            base_level = levels[0]
            base_cell_size = int(base_level["cell_size"])
            base_num_cells = int(base_level["num_cells"])

            for c in range(base_num_cells):
                qs = c * base_cell_size
                qe = min((c + 1) * base_cell_size, n)
                if qe <= qs:
                    continue

                q_slice = q_b[:, qs:qe, :]  # (H, Q, D)
                x_q = x_b[qs:qe, :]  # (Q, 3)

                near_start = max(0, c - span)
                near_end = min(base_num_cells, c + span + 1)
                ks = near_start * base_cell_size
                ke = min(n, near_end * base_cell_size)

                k_local = k_b[:, ks:ke, :]  # (H, K, D)
                v_local = v_b[:, ks:ke, :]  # (H, K, D)
                x_k = x_b[ks:ke, :]  # (K, 3)
                rel_local = x_k.unsqueeze(0) - x_q.unsqueeze(1)  # (Q, K, 3)
                dist_local = torch.linalg.norm(rel_local, dim=-1).clamp(min=1e-6)  # (Q, K)
                logits_local = torch.matmul(q_slice, k_local.transpose(-2, -1)) / sqrt_d
                logits_local = logits_local + self._dist_bias(dist_local)

                logits_parts = [logits_local]
                value_parts = [v_local]
                rel_parts = [rel_local]
                part_lengths = [logits_local.shape[-1]]
                q_center = x_q.mean(dim=0, keepdim=True)  # (1, 3)

                for level_idx in range(1, len(levels)):
                    lvl = levels[level_idx]
                    num_cells = int(lvl["num_cells"])
                    if num_cells <= 0:
                        continue
                    group = 2**level_idx
                    cell_idx = c // group
                    lvl_near_start = max(0, cell_idx - span)
                    lvl_near_end = min(num_cells, cell_idx + span + 1)
                    if lvl_near_start == 0 and lvl_near_end == num_cells:
                        continue

                    far_idx = torch.cat(
                        [
                            torch.arange(0, lvl_near_start, device=q.device, dtype=torch.long),
                            torch.arange(lvl_near_end, num_cells, device=q.device, dtype=torch.long),
                        ]
                    )
                    if far_idx.numel() == 0:
                        continue

                    centers = lvl["centers"]
                    assert isinstance(centers, torch.Tensor)
                    far_centers = centers[far_idx]
                    far_dist_to_q = torch.linalg.norm(far_centers - q_center, dim=-1)
                    if far_idx.numel() > self.fmm_far_topk:
                        _, topk_sel = torch.topk(
                            far_dist_to_q,
                            k=self.fmm_far_topk,
                            largest=False,
                        )
                        far_idx = far_idx[topk_sel]
                        far_centers = far_centers[topk_sel]

                    k_lvl = lvl["k"]
                    v_lvl = lvl["v"]
                    assert isinstance(k_lvl, torch.Tensor)
                    assert isinstance(v_lvl, torch.Tensor)
                    k_far = k_lvl[:, far_idx, :]  # (H, C_far, D)
                    v_far = v_lvl[:, far_idx, :]  # (H, C_far, D)
                    rel_far = far_centers.unsqueeze(0) - x_q.unsqueeze(1)  # (Q, C_far, 3)
                    dist_far = torch.linalg.norm(rel_far, dim=-1).clamp(min=1e-6)  # (Q, C_far)
                    logits_far = torch.matmul(q_slice, k_far.transpose(-2, -1)) / sqrt_d
                    logits_far = logits_far + self._dist_bias(dist_far)

                    logits_parts.append(logits_far)
                    value_parts.append(v_far)
                    rel_parts.append(rel_far)
                    part_lengths.append(logits_far.shape[-1])

                logits_all = torch.cat(logits_parts, dim=-1)  # (H, Q, K_total)
                attn_all = torch.softmax(logits_all, dim=-1)

                msg = torch.zeros(
                    self.num_heads,
                    qe - qs,
                    self.head_dim,
                    device=q.device,
                    dtype=q.dtype,
                )
                coord_delta = torch.zeros(qe - qs, 3, device=coords.device, dtype=coords.dtype)
                start = 0
                for part_len, v_part, rel_part in zip(part_lengths, value_parts, rel_parts):
                    end = start + part_len
                    attn_part = attn_all[..., start:end]
                    msg = msg + torch.matmul(attn_part, v_part)
                    coord_delta = coord_delta + (attn_part.mean(dim=0).unsqueeze(-1) * rel_part).sum(dim=1)
                    start = end

                target_idx = ordered_idx[qs:qe]
                msg_out[b, target_idx, :] = msg.transpose(0, 1).reshape(qe - qs, self.hidden_dim)
                coord_delta_out[b, target_idx, :] = coord_delta

        return msg_out, coord_delta_out

    def forward(
        self, hidden: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden: (B, L, C), coords: (B, L, 3), mask: (B, L)
        bsz, seq_len, _ = hidden.shape
        mask_f = mask.float()

        h = self.h_norm(hidden)
        q = self.to_q(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,L,D)
        k = self.to_k(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        valid_lengths = mask.sum(dim=1)
        use_fmm = self.use_fast_multipole and int(valid_lengths.max().item()) >= self.fmm_exact_threshold
        if use_fmm:
            msg, coord_delta = self._fmm_message_and_delta(q, k, v, coords, mask)
        else:
            msg, coord_delta = self._exact_message_and_delta(q, k, v, coords, mask)

        hidden = hidden + self.dropout(self.to_out(msg))
        hidden = hidden + self.dropout(self.ff(self.ff_norm(hidden)))
        hidden = hidden * mask_f.unsqueeze(-1)

        # Predict a coordinate update (delta) from the current state.
        # The delta uses only scalar weights and relative vectors -> equivariant.
        node_gate = self.coord_gate(hidden)  # (B, L, 1)
        coord_update = self.coord_step_size * node_gate * coord_delta
        coord_update = coord_update * mask_f.unsqueeze(-1)

        return hidden, coord_update


class SE3FoldingTransformer(nn.Module):
    """Iterative RNA 3D refinement model with configurable recycling passes."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        residue_vocab_size: int = 5,
        max_chain_embeddings: int = 64,
        max_copy_embeddings: int = 64,
        recycling_passes: int = 24,
        coord_step_size: float = 0.25,
        use_fast_multipole: bool = True,
        fmm_cell_size: int = 64,
        fmm_near_cell_span: int = 1,
        fmm_exact_threshold: int = 768,
        fmm_max_levels: int = 4,
        fmm_far_topk: int = 32,
        fmm_dynamic_reordering: bool = True,
    ) -> None:
        super().__init__()
        self.recycling_passes = recycling_passes

        self.residue_emb = nn.Embedding(residue_vocab_size, hidden_dim)
        self.chain_emb = nn.Embedding(max_chain_embeddings, hidden_dim)
        self.copy_emb = nn.Embedding(max_copy_embeddings, hidden_dim)
        self.resid_proj = nn.Linear(1, hidden_dim)
        self.coord_proj = nn.Linear(3, hidden_dim)
        self.template_proj = nn.Linear(3, hidden_dim)
        self.template_mix_logit = nn.Parameter(torch.tensor(0.0))

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [
                SE3RefinementBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    coord_step_size=coord_step_size,
                    use_fast_multipole=use_fast_multipole,
                    fmm_cell_size=fmm_cell_size,
                    fmm_near_cell_span=fmm_near_cell_span,
                    fmm_exact_threshold=fmm_exact_threshold,
                    fmm_max_levels=fmm_max_levels,
                    fmm_far_topk=fmm_far_topk,
                    fmm_dynamic_reordering=fmm_dynamic_reordering,
                )
                for _ in range(num_layers)
            ]
        )

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
        recycling_passes: Optional[int] = None,
    ) -> torch.Tensor:
        """Refine coordinates.

        Args:
            residue_idx: (B, L) residue token ids.
            chain_idx: (B, L) chain token ids.
            copy_idx: (B, L) copy ids.
            resid: (B, L) normalized residue indices.
            coords: (B, L, 3) input coordinates.
            mask: (B, L) valid-token mask.
            template_coords: (B, L, 3) MSA/template coordinates (optional).
            template_mask: (B, L) template availability mask (optional).
            recycling_passes: optional override of default recycle count.
        Returns:
            Refined coordinates tensor with shape (B, L, 3).
        """
        residue_idx = residue_idx.clamp_(min=0, max=self.residue_emb.num_embeddings - 1)
        chain_idx = chain_idx.clamp_(min=0, max=self.chain_emb.num_embeddings - 1)
        copy_idx = copy_idx.clamp_(min=0, max=self.copy_emb.num_embeddings - 1)
        if template_coords is None:
            template_coords = torch.zeros_like(coords)
        if template_mask is None:
            template_mask = mask
        template_mask_f = (template_mask & mask).unsqueeze(-1).float()
        template_mix = torch.sigmoid(self.template_mix_logit)

        hidden = (
            self.residue_emb(residue_idx)
            + self.chain_emb(chain_idx)
            + self.copy_emb(copy_idx)
            + self.resid_proj(resid.unsqueeze(-1))
            + self.coord_proj(coords)
            + template_mix * self.template_proj(template_coords) * template_mask_f
        )
        hidden = self.input_norm(hidden)
        hidden = hidden * mask.unsqueeze(-1).float()

        refined = coords + template_mix * (template_coords - coords) * template_mask_f
        num_recycles = recycling_passes if recycling_passes is not None else self.recycling_passes
        for _ in range(num_recycles):
            for layer in self.layers:
                hidden, coord_update = layer(hidden, refined, mask)
                refined = refined + coord_update
                hidden = hidden + self.coord_proj(coord_update)
                hidden = hidden * mask.unsqueeze(-1).float()

        return refined


if __name__ == "__main__":
    _ = SE3FoldingTransformer()
