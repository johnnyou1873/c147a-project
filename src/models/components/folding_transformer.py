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
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.coord_step_size = coord_step_size

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

        rel = coords.unsqueeze(1) - coords.unsqueeze(2)  # (B, L, L, 3): x_j - x_i at [i,j]
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True).clamp(min=1e-6)  # (B, L, L, 1)
        dist_bias = self.dist_bias(dist).permute(0, 3, 1, 2)  # (B, H, L, L)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_logits = attn_logits + dist_bias

        key_mask = mask[:, None, None, :]  # (B,1,1,L)
        query_mask = mask[:, None, :, None]  # (B,1,L,1)
        attn_logits = attn_logits.masked_fill(~key_mask, -1e4)
        attn = torch.softmax(attn_logits, dim=-1) * query_mask
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        msg = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, self.hidden_dim)
        hidden = hidden + self.dropout(self.to_out(msg))
        hidden = hidden + self.dropout(self.ff(self.ff_norm(hidden)))
        hidden = hidden * mask_f.unsqueeze(-1)

        # Coordinate update uses only scalar weights and relative vectors -> equivariant.
        attn_mean = attn.mean(dim=1)  # (B, L, L)
        coord_delta = (attn_mean.unsqueeze(-1) * rel).sum(dim=2)  # (B, L, 3)
        node_gate = self.coord_gate(hidden)  # (B, L, 1)
        coords = coords + self.coord_step_size * node_gate * coord_delta
        coords = coords * mask_f.unsqueeze(-1)

        return hidden, coords


class SE3FoldingTransformer(nn.Module):
    """Iterative RNA 3D refinement model with configurable recycling passes."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        residue_vocab_size: int = 5,
        max_chain_embeddings: int = 64,
        max_copy_embeddings: int = 64,
        recycling_passes: int = 5,
        coord_step_size: float = 0.25,
    ) -> None:
        super().__init__()
        self.recycling_passes = recycling_passes

        self.residue_emb = nn.Embedding(residue_vocab_size, hidden_dim)
        self.chain_emb = nn.Embedding(max_chain_embeddings, hidden_dim)
        self.copy_emb = nn.Embedding(max_copy_embeddings, hidden_dim)
        self.resid_proj = nn.Linear(1, hidden_dim)
        self.coord_proj = nn.Linear(3, hidden_dim)

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [
                SE3RefinementBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    coord_step_size=coord_step_size,
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
            recycling_passes: optional override of default recycle count.
        Returns:
            Refined coordinates tensor with shape (B, L, 3).
        """
        residue_idx = residue_idx.clamp_(min=0, max=self.residue_emb.num_embeddings - 1)
        chain_idx = chain_idx.clamp_(min=0, max=self.chain_emb.num_embeddings - 1)
        copy_idx = copy_idx.clamp_(min=0, max=self.copy_emb.num_embeddings - 1)

        hidden = (
            self.residue_emb(residue_idx)
            + self.chain_emb(chain_idx)
            + self.copy_emb(copy_idx)
            + self.resid_proj(resid.unsqueeze(-1))
            + self.coord_proj(coords)
        )
        hidden = self.input_norm(hidden)
        hidden = hidden * mask.unsqueeze(-1).float()

        refined = coords
        num_recycles = recycling_passes if recycling_passes is not None else self.recycling_passes
        for _ in range(num_recycles):
            for layer in self.layers:
                hidden, refined = layer(hidden, refined, mask)

        return refined


if __name__ == "__main__":
    _ = SE3FoldingTransformer()
