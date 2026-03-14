from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _mask_f(mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    return mask.unsqueeze(-1).to(dtype=dtype)


def _pair_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask[:, :, None] & mask[:, None, :]


def _pair_mask_f(mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    return _pair_mask(mask).unsqueeze(-1).to(dtype=dtype)


def _resolve_num_heads(width: int, requested: int) -> int:
    width = max(1, int(width))
    requested = max(1, int(requested))
    for candidate in range(min(width, requested), 0, -1):
        if width % candidate == 0:
            return candidate
    return 1


def _build_chain_chunk_ids(
    chain_idx: torch.Tensor,
    mask: torch.Tensor,
    target_chunk_size: int,
) -> torch.Tensor:
    chunk_ids = torch.full_like(chain_idx, fill_value=-1)
    target_chunk_size = max(1, int(target_chunk_size))
    for batch_idx in range(chain_idx.shape[0]):
        valid_positions = torch.nonzero(mask[batch_idx], as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            continue

        next_chunk = 0
        start = 0
        while start < valid_positions.numel():
            chain_value = int(chain_idx[batch_idx, valid_positions[start]].item())
            end = start
            while end < valid_positions.numel():
                pos = int(valid_positions[end].item())
                if int(chain_idx[batch_idx, pos].item()) != chain_value:
                    break
                end += 1

            chain_positions = valid_positions[start:end]
            chain_len = int(chain_positions.numel())
            chain_chunks = max(1, math.ceil(chain_len / target_chunk_size))
            boundaries = torch.linspace(
                0,
                chain_len,
                steps=chain_chunks + 1,
                device=chain_positions.device,
            ).round().to(dtype=torch.long)
            for chunk_offset in range(chain_chunks):
                lo = int(boundaries[chunk_offset].item())
                hi = int(boundaries[chunk_offset + 1].item())
                if hi <= lo:
                    continue
                chunk_ids[batch_idx, chain_positions[lo:hi]] = next_chunk
                next_chunk += 1

            start = end
    return chunk_ids


class LayerNorm(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        eps: float = 1e-5,
        create_scale: bool = True,
        create_offset: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)
        if create_scale:
            self.weight = nn.Parameter(torch.ones(self.hidden_dim))
        else:
            self.register_parameter("weight", None)
        if create_offset:
            self.bias = nn.Parameter(torch.zeros(self.hidden_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.hidden_dim,), self.weight, self.bias, self.eps)


class LinearNoBias(nn.Linear):
    def __init__(self, in_features: int, out_features: int, initializer: str = "default") -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=False)
        init = str(initializer).strip().lower()
        if init == "zeros":
            nn.init.zeros_(self.weight)
        elif init == "relu":
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0), nonlinearity="relu")
        else:
            nn.init.xavier_uniform_(self.weight)


class BiasInitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        biasinit: float = 0.0,
    ) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, float(biasinit))


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, c_a: int, c_s: int) -> None:
        super().__init__()
        self.layernorm_a = LayerNorm(c_a, create_scale=False, create_offset=False)
        self.layernorm_s = LayerNorm(c_s, create_offset=False)
        self.linear_s = nn.Linear(c_s, c_a)
        self.linear_nobias_s = LinearNoBias(c_s, c_a, initializer="zeros")
        nn.init.zeros_(self.linear_s.weight)
        nn.init.ones_(self.linear_s.bias)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        a = self.layernorm_a(a)
        s = self.layernorm_s(s)
        return torch.sigmoid(self.linear_s(s)) * a + self.linear_nobias_s(s)


class Transition(nn.Module):
    def __init__(self, c_in: int, n: int) -> None:
        super().__init__()
        self.layernorm = LayerNorm(c_in)
        self.linear_a = LinearNoBias(c_in, n * c_in, initializer="relu")
        self.linear_b = LinearNoBias(c_in, n * c_in, initializer="relu")
        self.linear_out = LinearNoBias(n * c_in, c_in, initializer="zeros")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x)
        a = self.linear_a(x)
        b = self.linear_b(x)
        return self.linear_out(F.silu(a) * b)


class FourierEmbedding(nn.Module):
    def __init__(self, c: int, seed: int = 42) -> None:
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        self.register_buffer("w", torch.randn(c, generator=generator), persistent=False)
        self.register_buffer("b", torch.randn(c, generator=generator), persistent=False)

    def forward(self, noise_level: torch.Tensor) -> torch.Tensor:
        return torch.cos(2.0 * torch.pi * (noise_level.unsqueeze(-1) * self.w + self.b))


class AttentionPairBias(nn.Module):
    def __init__(
        self,
        *,
        has_s: bool,
        n_heads: int,
        c_a: int,
        c_z: int,
        c_s: int = 0,
        biasinit: float = -2.0,
    ) -> None:
        super().__init__()
        self.has_s = bool(has_s)
        self.c_a = int(c_a)
        self.c_z = int(c_z)
        self.n_heads = _resolve_num_heads(self.c_a, n_heads)
        self.head_dim = self.c_a // self.n_heads

        if self.has_s:
            self.layernorm_a: nn.Module = AdaptiveLayerNorm(c_a=self.c_a, c_s=c_s)
        else:
            self.layernorm_a = LayerNorm(self.c_a)

        self.q_proj = nn.Linear(self.c_a, self.c_a)
        self.k_proj = nn.Linear(self.c_a, self.c_a)
        self.v_proj = nn.Linear(self.c_a, self.c_a)
        self.g_proj = nn.Linear(self.c_a, self.c_a)
        self.out_proj = nn.Linear(self.c_a, self.c_a)
        self.layernorm_z = LayerNorm(self.c_z, create_offset=False)
        self.linear_nobias_z = LinearNoBias(self.c_z, self.n_heads)

        if self.has_s:
            self.linear_a_last = BiasInitLinear(c_s, self.c_a, bias=True, biasinit=biasinit)
        else:
            self.linear_a_last = None

    def forward(
        self,
        *,
        a: torch.Tensor,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = a.shape
        if self.has_s:
            if s is None:
                raise ValueError("AttentionPairBias with has_s=True requires `s`.")
            a_norm = self.layernorm_a(a, s)
        else:
            a_norm = self.layernorm_a(a)

        q = self.q_proj(a_norm).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(a_norm).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(a_norm).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        bias = self.linear_nobias_z(self.layernorm_z(z)).permute(0, 3, 1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        logits = logits + bias
        if mask is not None:
            logits = logits.masked_fill(~_pair_mask(mask).unsqueeze(1), -1e4)

        attn = torch.softmax(logits, dim=-1)
        if mask is not None:
            attn = attn * mask[:, None, None, :].to(dtype=attn.dtype)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, self.c_a)
        out = torch.sigmoid(self.g_proj(a_norm)) * out
        out = self.out_proj(out)
        if self.linear_a_last is not None and s is not None:
            out = torch.sigmoid(self.linear_a_last(s)) * out
        if mask is not None:
            out = out * _mask_f(mask, dtype=out.dtype)
        return out


class TriangleMultiplication(nn.Module):
    def __init__(self, c_z: int, *, outgoing: bool, use_chunking: bool = False) -> None:
        super().__init__()
        self.outgoing = bool(outgoing)
        self.use_chunking = bool(use_chunking)
        self.layernorm = LayerNorm(c_z)
        self.left_proj = LinearNoBias(c_z, c_z, initializer="relu")
        self.right_proj = LinearNoBias(c_z, c_z, initializer="relu")
        self.left_gate = nn.Linear(c_z, c_z)
        self.right_gate = nn.Linear(c_z, c_z)
        self.output_norm = LayerNorm(c_z)
        self.output_proj = LinearNoBias(c_z, c_z, initializer="zeros")
        self.output_gate = nn.Linear(c_z, c_z)

    def _chunked_outgoing(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        chunk_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _, channels = left.shape
        max_chunks = int(chunk_ids.max().item()) + 1 if bool((chunk_ids >= 0).any().item()) else 0
        if max_chunks <= 0:
            return left.new_zeros(batch, seq_len, seq_len, channels)

        pooled_left = left.new_zeros(batch, seq_len, max_chunks, channels)
        pooled_right = right.new_zeros(batch, seq_len, max_chunks, channels)
        chunk_counts = left.new_zeros(batch, max_chunks)
        for batch_idx in range(batch):
            valid = chunk_ids[batch_idx] >= 0
            if not bool(valid.any().item()):
                continue
            active_ids = chunk_ids[batch_idx, valid]
            ones = torch.ones_like(active_ids, dtype=left.dtype)
            chunk_counts[batch_idx].index_add_(0, active_ids, ones)

            gather_index = active_ids.view(1, -1, 1).expand(seq_len, -1, channels)
            pooled_left[batch_idx].scatter_add_(1, gather_index, left[batch_idx, :, valid, :])
            pooled_right[batch_idx].scatter_add_(1, gather_index, right[batch_idx, :, valid, :])

        denom = chunk_counts.clamp(min=1.0).view(batch, 1, max_chunks, 1)
        pooled_left = pooled_left / denom
        pooled_right = pooled_right / denom
        return torch.einsum("birc,bjrc,br->bijc", pooled_left, pooled_right, chunk_counts)

    def _chunked_incoming(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        chunk_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, seq_len, channels = left.shape
        max_chunks = int(chunk_ids.max().item()) + 1 if bool((chunk_ids >= 0).any().item()) else 0
        if max_chunks <= 0:
            return left.new_zeros(batch, seq_len, seq_len, channels)

        pooled_left = left.new_zeros(batch, max_chunks, seq_len, channels)
        pooled_right = right.new_zeros(batch, max_chunks, seq_len, channels)
        chunk_counts = left.new_zeros(batch, max_chunks)
        for batch_idx in range(batch):
            valid = chunk_ids[batch_idx] >= 0
            if not bool(valid.any().item()):
                continue
            active_ids = chunk_ids[batch_idx, valid]
            ones = torch.ones_like(active_ids, dtype=left.dtype)
            chunk_counts[batch_idx].index_add_(0, active_ids, ones)

            gather_index = active_ids.view(-1, 1, 1).expand(-1, seq_len, channels)
            pooled_left[batch_idx].scatter_add_(0, gather_index, left[batch_idx, valid, :, :])
            pooled_right[batch_idx].scatter_add_(0, gather_index, right[batch_idx, valid, :, :])

        denom = chunk_counts.clamp(min=1.0).view(batch, max_chunks, 1, 1)
        pooled_left = pooled_left / denom
        pooled_right = pooled_right / denom
        return torch.einsum("bric,brjc,br->bijc", pooled_left, pooled_right, chunk_counts)

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z_norm = self.layernorm(z)
        pair_mask_f = pair_mask.unsqueeze(-1).to(dtype=z.dtype)
        left = self.left_proj(z_norm) * torch.sigmoid(self.left_gate(z_norm))
        right = self.right_proj(z_norm) * torch.sigmoid(self.right_gate(z_norm))
        left = left * pair_mask_f
        right = right * pair_mask_f
        if self.use_chunking and chunk_ids is not None:
            if self.outgoing:
                update = self._chunked_outgoing(left, right, chunk_ids)
            else:
                update = self._chunked_incoming(left, right, chunk_ids)
        elif self.outgoing:
            update = torch.einsum("bikc,bjkc->bijc", left, right)
        else:
            update = torch.einsum("bkic,bkjc->bijc", left, right)
        update = update / math.sqrt(max(1, z.shape[1]))
        update = self.output_proj(self.output_norm(update))
        update = update * torch.sigmoid(self.output_gate(z_norm))
        return update * pair_mask_f


class TriangleAttention(nn.Module):
    def __init__(self, c_z: int, no_heads: int = 4) -> None:
        super().__init__()
        self.c_z = int(c_z)
        self.no_heads = _resolve_num_heads(self.c_z, no_heads)
        self.head_dim = self.c_z // self.no_heads
        self.layernorm = LayerNorm(self.c_z)
        self.q_proj = nn.Linear(self.c_z, self.c_z)
        self.k_proj = nn.Linear(self.c_z, self.c_z)
        self.v_proj = nn.Linear(self.c_z, self.c_z)
        self.bias_proj = LinearNoBias(self.c_z, self.no_heads)
        self.out_proj = nn.Linear(self.c_z, self.c_z)

    def forward(self, z: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _, _ = z.shape
        z_norm = self.layernorm(z).reshape(bsz * seq_len, seq_len, self.c_z)
        row_mask = pair_mask.reshape(bsz * seq_len, seq_len)

        q = self.q_proj(z_norm).view(bsz * seq_len, seq_len, self.no_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(z_norm).view(bsz * seq_len, seq_len, self.no_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(z_norm).view(bsz * seq_len, seq_len, self.no_heads, self.head_dim).transpose(1, 2)
        bias = self.bias_proj(z_norm).permute(0, 2, 1).unsqueeze(-2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        logits = logits + bias
        logits = logits.masked_fill(~row_mask[:, None, None, :], -1e4)

        attn = torch.softmax(logits, dim=-1)
        attn = attn * row_mask[:, None, :, None].to(dtype=attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz * seq_len, seq_len, self.c_z)
        out = self.out_proj(out).reshape(bsz, seq_len, seq_len, self.c_z)
        return out * pair_mask.unsqueeze(-1).to(dtype=out.dtype)


class TriangleLinearAttention(nn.Module):
    def __init__(self, c_z: int, no_heads: int = 4, feature_map_multiplier: int = 2) -> None:
        super().__init__()
        self.c_z = int(c_z)
        self.no_heads = _resolve_num_heads(self.c_z, no_heads)
        self.head_dim = self.c_z // self.no_heads
        self.feature_dim = max(2, int(feature_map_multiplier) * self.head_dim)
        if self.feature_dim % 2 != 0:
            self.feature_dim += 1
        self.base_feature_dim = self.feature_dim // 2

        self.layernorm = LayerNorm(self.c_z)
        self.q_proj = nn.Linear(self.c_z, self.c_z)
        self.k_proj = nn.Linear(self.c_z, self.c_z)
        self.v_proj = nn.Linear(self.c_z, self.c_z)
        self.q_feature = nn.Linear(self.head_dim, self.base_feature_dim)
        self.k_feature = nn.Linear(self.head_dim, self.base_feature_dim)
        self.gate_proj = nn.Linear(self.c_z, self.c_z)
        self.out_proj = nn.Linear(self.c_z, self.c_z)

    def _feature_map(self, tensor: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        mapped = projection(tensor)
        mapped = mapped.clamp(min=-8.0, max=8.0)
        return torch.cat([torch.exp(mapped), torch.exp(-mapped)], dim=-1)

    def forward(self, z: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _, _ = z.shape
        z_rows = self.layernorm(z).reshape(bsz * seq_len, seq_len, self.c_z)
        row_mask = pair_mask.reshape(bsz * seq_len, seq_len)
        row_mask_f = row_mask.to(dtype=z.dtype).unsqueeze(1).unsqueeze(-1)

        q = self.q_proj(z_rows).view(bsz * seq_len, seq_len, self.no_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(z_rows).view(bsz * seq_len, seq_len, self.no_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(z_rows).view(bsz * seq_len, seq_len, self.no_heads, self.head_dim).transpose(1, 2)

        q_features = self._feature_map(q, self.q_feature)
        k_features = self._feature_map(k, self.k_feature) * row_mask_f
        v = v * row_mask_f

        kv = torch.einsum("bhnd,bhne->bhde", k_features, v)
        k_sum = k_features.sum(dim=-2)
        denom = torch.einsum("bhnd,bhd->bhn", q_features, k_sum).clamp(min=1e-6)
        out = torch.einsum("bhnd,bhde->bhne", q_features, kv)
        out = out / denom.unsqueeze(-1)
        out = out.transpose(1, 2).reshape(bsz * seq_len, seq_len, self.c_z)
        out = torch.sigmoid(self.gate_proj(z_rows)) * out
        out = self.out_proj(out).reshape(bsz, seq_len, seq_len, self.c_z)
        return out * pair_mask.unsqueeze(-1).to(dtype=out.dtype)


class InputFeatureEmbedder(nn.Module):
    def __init__(
        self,
        *,
        residue_vocab_size: int,
        max_chain_embeddings: int,
        max_copy_embeddings: int,
        c_token: int,
        restype_dim: int = 32,
        profile_dim: int = 32,
    ) -> None:
        super().__init__()
        self.residue_vocab_size = int(residue_vocab_size)
        self.c_token = int(c_token)
        self.restype_dim = int(restype_dim)
        self.profile_dim = int(profile_dim)
        self.c_s_inputs = self.c_token + self.restype_dim + self.profile_dim + 1

        self.residue_emb = nn.Embedding(self.residue_vocab_size, self.c_token)
        self.chain_emb = nn.Embedding(max_chain_embeddings, self.c_token)
        self.copy_emb = nn.Embedding(max_copy_embeddings, self.c_token)
        self.resid_proj = nn.Linear(1, self.c_token)
        self.coord_proj = nn.Linear(3, self.c_token)
        self.local_geom_proj = nn.Sequential(
            nn.Linear(5, self.c_token),
            nn.GELU(),
            LinearNoBias(self.c_token, self.c_token),
        )
        self.restype_proj = LinearNoBias(self.residue_vocab_size, self.restype_dim)
        self.profile_proj = LinearNoBias(self.residue_vocab_size, self.profile_dim)
        self.token_transition = Transition(self.c_token, n=2)

    def _summarize_local_geometry(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        rna_bpp_banded: Optional[torch.Tensor],
        rna_bpp_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = coords.shape
        dtype = coords.dtype
        device = coords.device

        prev_dist = torch.zeros(bsz, seq_len, device=device, dtype=dtype)
        next_dist = torch.zeros(bsz, seq_len, device=device, dtype=dtype)
        twohop_dist = torch.zeros(bsz, seq_len, device=device, dtype=dtype)
        if seq_len > 1:
            d1 = torch.linalg.norm(coords[:, 1:] - coords[:, :-1], dim=-1)
            prev_dist[:, 1:] = d1
            next_dist[:, :-1] = d1
        if seq_len > 2:
            d2 = torch.linalg.norm(coords[:, 2:] - coords[:, :-2], dim=-1)
            twohop_dist[:, 1:-1] = d2

        if rna_bpp_banded is not None:
            bpp = rna_bpp_banded.to(device=device, dtype=dtype)
            if rna_bpp_mask is not None:
                bpp = bpp * rna_bpp_mask.to(device=device, dtype=torch.bool).to(dtype=dtype)
            bpp_mass = bpp.sum(dim=-1)
            bpp_peak = bpp.max(dim=-1).values
        else:
            bpp_mass = torch.zeros(bsz, seq_len, device=device, dtype=dtype)
            bpp_peak = torch.zeros(bsz, seq_len, device=device, dtype=dtype)

        feat = torch.stack([prev_dist, next_dist, twohop_dist, bpp_mass, bpp_peak], dim=-1)
        return feat * _mask_f(mask, dtype=dtype)

    def forward(
        self,
        *,
        residue_idx: torch.Tensor,
        chain_idx: torch.Tensor,
        copy_idx: torch.Tensor,
        resid: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        rna_msa_profile: Optional[torch.Tensor],
        rna_bpp_banded: Optional[torch.Tensor],
        rna_bpp_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residue_one_hot = F.one_hot(
            residue_idx.clamp(min=0, max=self.residue_vocab_size - 1),
            num_classes=self.residue_vocab_size,
        ).to(dtype=coords.dtype)
        profile = residue_one_hot if rna_msa_profile is None else rna_msa_profile.to(dtype=coords.dtype)

        token = (
            self.residue_emb(residue_idx)
            + self.chain_emb(chain_idx)
            + self.copy_emb(copy_idx)
            + self.resid_proj(resid.unsqueeze(-1).to(dtype=coords.dtype))
            + self.coord_proj(coords)
            + self.local_geom_proj(
                self._summarize_local_geometry(
                    coords=coords,
                    mask=mask,
                    rna_bpp_banded=rna_bpp_banded,
                    rna_bpp_mask=rna_bpp_mask,
                )
            )
        )
        token = token + self.token_transition(token)
        token = token * _mask_f(mask, dtype=coords.dtype)

        restype = self.restype_proj(residue_one_hot) * _mask_f(mask, dtype=coords.dtype)
        profile_embed = self.profile_proj(profile) * _mask_f(mask, dtype=coords.dtype)
        deletion_mean = torch.zeros(
            residue_idx.shape[0],
            residue_idx.shape[1],
            1,
            device=coords.device,
            dtype=coords.dtype,
        )
        return torch.cat([token, restype, profile_embed, deletion_mean], dim=-1)


class RelativePositionEncoding(nn.Module):
    def __init__(self, c_z: int, r_max: int = 32, s_max: int = 2) -> None:
        super().__init__()
        self.c_z = int(c_z)
        self.r_max = max(1, int(r_max))
        self.s_max = max(1, int(s_max))
        feat_dim = 2 * (self.r_max + 1) + 2 * (self.r_max + 1) + 1 + 2 * (self.s_max + 1)
        self.linear_no_bias = LinearNoBias(feat_dim, self.c_z)

    def build_feature(self, chain_idx: torch.Tensor, copy_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = chain_idx.shape
        device = mask.device
        token_index = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)

        same_chain = chain_idx[:, :, None] == chain_idx[:, None, :]
        same_entity = same_chain

        residue_delta = token_index[:, :, None] - token_index[:, None, :]
        d_residue = residue_delta.clamp(min=-self.r_max, max=self.r_max) + self.r_max
        d_residue = torch.where(same_chain, d_residue, d_residue.new_full((), 2 * self.r_max + 1))
        a_rel_pos = F.one_hot(d_residue, num_classes=2 * (self.r_max + 1))

        d_token = residue_delta.clamp(min=-self.r_max, max=self.r_max) + self.r_max
        d_token = torch.where(same_chain, d_token, d_token.new_full((), 2 * self.r_max + 1))
        a_rel_token = F.one_hot(d_token, num_classes=2 * (self.r_max + 1))

        copy_delta = (copy_idx[:, :, None] - copy_idx[:, None, :]).clamp(min=-self.s_max, max=self.s_max) + self.s_max
        d_chain = torch.where(same_entity, copy_delta, copy_delta.new_full((), 2 * self.s_max + 1))
        a_rel_chain = F.one_hot(d_chain, num_classes=2 * (self.s_max + 1))

        relp = torch.cat(
            [
                a_rel_pos.to(dtype=torch.float32),
                a_rel_token.to(dtype=torch.float32),
                same_entity.unsqueeze(-1).to(dtype=torch.float32),
                a_rel_chain.to(dtype=torch.float32),
            ],
            dim=-1,
        )
        return relp * _pair_mask_f(mask, dtype=relp.dtype)

    def forward(self, relp_feature: torch.Tensor) -> torch.Tensor:
        return self.linear_no_bias(relp_feature)


class PairformerBlock(nn.Module):
    def __init__(
        self,
        *,
        c_z: int,
        c_s: int,
        n_heads: int = 8,
        no_heads_pair: int = 4,
        num_intermediate_factor: int = 4,
        dropout: float = 0.25,
        triangle_attention_mode: str = "linear",
        triangle_feature_map_multiplier: int = 2,
        use_chunked_triangle_multiplication: bool = False,
    ) -> None:
        super().__init__()
        self.c_s = int(c_s)
        self.dropout = float(dropout)
        self.use_chunked_triangle_multiplication = bool(use_chunked_triangle_multiplication)
        attention_mode = str(triangle_attention_mode).strip().lower().replace("-", "_")
        if attention_mode == "linear":
            triangle_attention_cls = TriangleLinearAttention
        elif attention_mode in {"softmax", "cubic", "standard"}:
            triangle_attention_cls = TriangleAttention
        else:
            raise ValueError(
                f"Unsupported triangle_attention_mode='{triangle_attention_mode}'. Expected 'linear' or 'softmax'."
            )

        self.tri_mul_out = TriangleMultiplication(
            c_z,
            outgoing=True,
            use_chunking=self.use_chunked_triangle_multiplication,
        )
        self.tri_mul_in = TriangleMultiplication(
            c_z,
            outgoing=False,
            use_chunking=self.use_chunked_triangle_multiplication,
        )
        self.tri_att_start = triangle_attention_cls(
            c_z,
            no_heads=no_heads_pair,
            **({"feature_map_multiplier": triangle_feature_map_multiplier} if triangle_attention_cls is TriangleLinearAttention else {}),
        )
        self.tri_att_end = triangle_attention_cls(
            c_z,
            no_heads=no_heads_pair,
            **({"feature_map_multiplier": triangle_feature_map_multiplier} if triangle_attention_cls is TriangleLinearAttention else {}),
        )
        self.pair_transition = Transition(c_z, n=num_intermediate_factor)
        if self.c_s > 0:
            self.attention_pair_bias = AttentionPairBias(
                has_s=False,
                n_heads=n_heads,
                c_a=self.c_s,
                c_z=c_z,
            )
            self.single_transition = Transition(self.c_s, n=4)
        else:
            self.attention_pair_bias = None
            self.single_transition = None

    def forward(
        self,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        mask: Optional[torch.Tensor],
        chunk_ids: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        pair_mask_f = pair_mask.unsqueeze(-1).to(dtype=z.dtype)
        z = z + F.dropout(self.tri_mul_out(z, pair_mask, chunk_ids=chunk_ids), p=self.dropout, training=self.training)
        z = z + F.dropout(self.tri_mul_in(z, pair_mask, chunk_ids=chunk_ids), p=self.dropout, training=self.training)
        z = z + F.dropout(self.tri_att_start(z, pair_mask), p=self.dropout, training=self.training)
        z_t = z.transpose(1, 2).contiguous()
        z_t = z_t + F.dropout(
            self.tri_att_end(z_t, pair_mask.transpose(-1, -2)),
            p=self.dropout,
            training=self.training,
        )
        z = z_t.transpose(1, 2).contiguous()
        z = z + self.pair_transition(z)
        z = 0.5 * (z + z.transpose(1, 2))
        z = z * pair_mask_f

        if self.c_s > 0 and s is not None and self.attention_pair_bias is not None and self.single_transition is not None:
            s = s + self.attention_pair_bias(a=s, s=None, z=z, mask=mask)
            s = s + self.single_transition(s)
            if mask is not None:
                s = s * _mask_f(mask, dtype=s.dtype)
        return s, z


class PairformerStack(nn.Module):
    def __init__(
        self,
        *,
        n_blocks: int,
        c_z: int,
        c_s: int,
        n_heads: int = 8,
        dropout: float = 0.25,
        triangle_attention_mode: str = "linear",
        triangle_feature_map_multiplier: int = 2,
        use_chunked_triangle_multiplication: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                PairformerBlock(
                    c_z=c_z,
                    c_s=c_s,
                    n_heads=n_heads,
                    no_heads_pair=max(1, n_heads // 2),
                    dropout=dropout,
                    triangle_attention_mode=triangle_attention_mode,
                    triangle_feature_map_multiplier=triangle_feature_map_multiplier,
                    use_chunked_triangle_multiplication=use_chunked_triangle_multiplication,
                )
                for _ in range(max(0, int(n_blocks)))
            ]
        )

    def forward(
        self,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        mask: Optional[torch.Tensor],
        chunk_ids: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        for block in self.blocks:
            s, z = block(s=s, z=z, pair_mask=pair_mask, mask=mask, chunk_ids=chunk_ids)
        return s, z


class OuterProductMean(nn.Module):
    def __init__(self, c_m: int, c_z: int, c_hidden: int) -> None:
        super().__init__()
        self.layernorm_m = LayerNorm(c_m)
        self.linear_left = LinearNoBias(c_m, c_hidden, initializer="relu")
        self.linear_right = LinearNoBias(c_m, c_hidden, initializer="relu")
        self.layernorm_out = LayerNorm(c_hidden * c_hidden)
        self.linear_out = LinearNoBias(c_hidden * c_hidden, c_z, initializer="zeros")

    def forward(self, m: torch.Tensor, msa_mask: torch.Tensor) -> torch.Tensor:
        m = self.layernorm_m(m)
        mask_f = msa_mask.unsqueeze(-1).to(dtype=m.dtype)
        left = self.linear_left(m) * mask_f
        right = self.linear_right(m) * mask_f
        outer = torch.einsum("bric,brjd->bijcd", left, right)
        norm = torch.einsum(
            "bri,brj->bij",
            msa_mask.to(dtype=m.dtype),
            msa_mask.to(dtype=m.dtype),
        ).clamp(min=1.0)
        outer = outer / norm.unsqueeze(-1).unsqueeze(-1)
        outer = outer.reshape(*outer.shape[:3], -1)
        return self.linear_out(self.layernorm_out(outer))


class MSAPairWeightedAveraging(nn.Module):
    def __init__(self, c_m: int, c_z: int, c: int, n_heads: int = 8) -> None:
        super().__init__()
        self.c = int(c)
        self.n_heads = _resolve_num_heads(c_m, n_heads)
        self.layernorm_m = LayerNorm(c_m)
        self.linear_mv = LinearNoBias(c_m, self.c * self.n_heads)
        self.layernorm_z = LayerNorm(c_z)
        self.linear_z = LinearNoBias(c_z, self.n_heads)
        self.linear_mg = LinearNoBias(c_m, self.c * self.n_heads, initializer="zeros")
        self.linear_out = LinearNoBias(self.c * self.n_heads, c_m, initializer="zeros")

    def forward(self, m: torch.Tensor, z: torch.Tensor, msa_mask: torch.Tensor) -> torch.Tensor:
        m = m * msa_mask.unsqueeze(-1).to(dtype=m.dtype)
        m_norm = self.layernorm_m(m)
        v = self.linear_mv(m_norm).view(*m_norm.shape[:-1], self.n_heads, self.c)
        g = torch.sigmoid(self.linear_mg(m_norm)).view(*m_norm.shape[:-1], self.n_heads, self.c)
        bias = self.linear_z(self.layernorm_z(z))
        w = torch.softmax(bias, dim=-2)
        wv = torch.einsum("bijh,brjhc->brihc", w, v)
        out = self.linear_out((g * wv).reshape(*m_norm.shape[:-1], self.n_heads * self.c))
        return out * msa_mask.unsqueeze(-1).to(dtype=out.dtype)


class MSAStack(nn.Module):
    def __init__(self, c_m: int, c_z: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
            c_m=c_m,
            c_z=c_z,
            c=max(8, c_m // 4),
        )
        self.transition_m = Transition(c_m, n=4)

    def forward(self, m: torch.Tensor, z: torch.Tensor, msa_mask: torch.Tensor) -> torch.Tensor:
        m = m + F.dropout(
            self.msa_pair_weighted_averaging(m, z, msa_mask),
            p=self.dropout,
            training=self.training,
        )
        m = m + self.transition_m(m)
        return m * msa_mask.unsqueeze(-1).to(dtype=m.dtype)


class MSABlock(nn.Module):
    def __init__(
        self,
        *,
        c_m: int,
        c_z: int,
        is_last_block: bool,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.is_last_block = bool(is_last_block)
        self.outer_product_mean_msa = OuterProductMean(
            c_m=c_m,
            c_z=c_z,
            c_hidden=max(8, c_m // 4),
        )
        if not self.is_last_block:
            self.msa_stack = MSAStack(c_m=c_m, c_z=c_z, dropout=msa_dropout)
        else:
            self.msa_stack = None
        self.pair_stack = PairformerBlock(
            c_z=c_z,
            c_s=0,
            n_heads=max(1, c_z // 8),
            dropout=pair_dropout,
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        msa_mask: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        z = z + self.outer_product_mean_msa(m, msa_mask)
        if self.msa_stack is not None:
            m = self.msa_stack(m, z, msa_mask)
        _, z = self.pair_stack(s=None, z=z, pair_mask=pair_mask, mask=None)
        return (m, z) if self.msa_stack is not None else (None, z)


class MSAModule(nn.Module):
    def __init__(
        self,
        *,
        n_blocks: int,
        c_m: int,
        c_z: int,
        c_s_inputs: int,
        residue_vocab_size: int,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_blocks = max(0, int(n_blocks))
        self.residue_vocab_size = int(residue_vocab_size)
        self.linear_no_bias_m = LinearNoBias(self.residue_vocab_size + 2, c_m)
        self.linear_no_bias_s = LinearNoBias(c_s_inputs, c_m)
        self.blocks = nn.ModuleList(
            [
                MSABlock(
                    c_m=c_m,
                    c_z=c_z,
                    is_last_block=(i + 1 == self.n_blocks),
                    msa_dropout=msa_dropout,
                    pair_dropout=pair_dropout,
                )
                for i in range(self.n_blocks)
            ]
        )

    def forward(
        self,
        *,
        rna_msa_tokens: torch.Tensor,
        rna_msa_mask: torch.Tensor,
        rna_msa_row_valid: Optional[torch.Tensor],
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.n_blocks < 1:
            return z

        row_valid = (
            torch.ones(rna_msa_tokens.shape[:2], device=rna_msa_tokens.device, dtype=torch.bool)
            if rna_msa_row_valid is None
            else rna_msa_row_valid.to(device=rna_msa_tokens.device, dtype=torch.bool)
        )
        msa_mask = rna_msa_mask.to(dtype=torch.bool) & row_valid.unsqueeze(-1)
        tokens = rna_msa_tokens.clamp(min=0, max=self.residue_vocab_size - 1)
        msa_one_hot = F.one_hot(tokens, num_classes=self.residue_vocab_size).to(dtype=s_inputs.dtype)
        has_deletion = torch.zeros((*tokens.shape, 1), device=tokens.device, dtype=s_inputs.dtype)
        deletion_value = torch.zeros((*tokens.shape, 1), device=tokens.device, dtype=s_inputs.dtype)
        msa_feat = torch.cat([msa_one_hot, has_deletion, deletion_value], dim=-1)
        m = self.linear_no_bias_m(msa_feat) + self.linear_no_bias_s(s_inputs).unsqueeze(1)
        m = m * msa_mask.unsqueeze(-1).to(dtype=m.dtype)

        for block in self.blocks:
            m, z = block(m=m, z=z, pair_mask=pair_mask, msa_mask=msa_mask)
            if m is None:
                break
        return z * pair_mask.unsqueeze(-1).to(dtype=z.dtype)


class TemplateEmbedder(nn.Module):
    def __init__(
        self,
        *,
        n_blocks: int,
        c: int,
        c_z: int,
        residue_vocab_size: int,
        n_heads: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_blocks = max(0, int(n_blocks))
        self.c = int(c)
        self.c_z = int(c_z)
        self.residue_vocab_size = int(residue_vocab_size)
        self.no_bins = 39
        template_feature_dim = self.no_bins + 1 + 3 + 1 + 2 * self.residue_vocab_size

        self.layernorm_z = LayerNorm(self.c_z)
        self.linear_no_bias_z = LinearNoBias(self.c_z, self.c)
        self.linear_no_bias_a = LinearNoBias(template_feature_dim, self.c)
        self.pairformer_stack = PairformerStack(
            n_blocks=self.n_blocks,
            c_z=self.c,
            c_s=0,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.layernorm_v = LayerNorm(self.c)
        self.linear_no_bias_u = LinearNoBias(self.c, self.c_z)

    def _template_distogram(self, distances: torch.Tensor, pair_valid: torch.Tensor) -> torch.Tensor:
        edges = torch.linspace(
            3.25,
            50.75,
            steps=self.no_bins - 1,
            device=distances.device,
            dtype=distances.dtype,
        )
        bucket = torch.bucketize(distances, boundaries=edges)
        dgram = F.one_hot(bucket, num_classes=self.no_bins).to(dtype=distances.dtype)
        return dgram * pair_valid.unsqueeze(-1).to(dtype=distances.dtype)

    def _single_template_features(
        self,
        coords: torch.Tensor,
        template_mask: torch.Tensor,
        residue_idx: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        pair_valid = pair_mask & template_mask[:, :, None] & template_mask[:, None, :]
        pair_valid_f = pair_valid.unsqueeze(-1).to(dtype=coords.dtype)

        diffs = coords[:, :, None, :] - coords[:, None, :, :]
        distances = torch.linalg.norm(diffs, dim=-1)
        unit_vector = diffs / distances.unsqueeze(-1).clamp(min=1e-6)
        unit_vector = unit_vector * pair_valid_f
        distogram = self._template_distogram(distances, pair_valid)
        pseudo_beta_mask = pair_valid.unsqueeze(-1).to(dtype=coords.dtype)
        backbone_frame_mask = pair_valid.unsqueeze(-1).to(dtype=coords.dtype)

        residue_one_hot = F.one_hot(
            residue_idx.clamp(min=0, max=self.residue_vocab_size - 1),
            num_classes=self.residue_vocab_size,
        ).to(dtype=coords.dtype)
        restype_i = residue_one_hot[:, :, None, :].expand(-1, -1, coords.shape[1], -1)
        restype_j = residue_one_hot[:, None, :, :].expand(-1, coords.shape[1], -1, -1)
        features = torch.cat(
            [
                distogram,
                backbone_frame_mask,
                unit_vector,
                pseudo_beta_mask,
                restype_i,
                restype_j,
            ],
            dim=-1,
        )
        return features * pair_valid_f

    def forward(
        self,
        *,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        template_topk_coords: Optional[torch.Tensor],
        template_topk_mask: Optional[torch.Tensor],
        template_topk_valid: Optional[torch.Tensor],
        template_topk_residue_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.n_blocks < 1 or template_topk_coords is None or template_topk_coords.numel() == 0:
            return torch.zeros_like(z)

        bsz, n_templates, seq_len, _ = template_topk_coords.shape
        z_proj = self.linear_no_bias_z(self.layernorm_z(z))
        template_sum = torch.zeros_like(z_proj)
        template_count = torch.zeros(bsz, 1, 1, 1, device=z.device, dtype=z.dtype)

        if template_topk_mask is None:
            template_topk_mask = torch.ones(
                bsz,
                n_templates,
                seq_len,
                device=z.device,
                dtype=torch.bool,
            )
        if template_topk_valid is None:
            template_topk_valid = template_topk_mask.any(dim=-1)
        if template_topk_residue_idx is None:
            template_topk_residue_idx = torch.zeros(
                bsz,
                n_templates,
                seq_len,
                device=z.device,
                dtype=torch.long,
            )

        for template_id in range(n_templates):
            active = template_topk_valid[:, template_id].to(device=z.device, dtype=torch.bool)
            if not bool(active.any().item()):
                continue
            template_features = self._single_template_features(
                coords=template_topk_coords[:, template_id].to(device=z.device, dtype=z.dtype),
                template_mask=template_topk_mask[:, template_id].to(device=z.device, dtype=torch.bool),
                residue_idx=template_topk_residue_idx[:, template_id].to(device=z.device, dtype=torch.long),
                pair_mask=pair_mask,
            )
            v = z_proj + self.linear_no_bias_a(template_features)
            _, v = self.pairformer_stack(s=None, z=v, pair_mask=pair_mask, mask=None)
            v = self.layernorm_v(v)
            active_f = active[:, None, None, None].to(dtype=v.dtype)
            template_sum = template_sum + active_f * v
            template_count = template_count + active_f

        template_mean = template_sum / template_count.clamp(min=1.0)
        return self.linear_no_bias_u(F.relu(template_mean)) * pair_mask.unsqueeze(-1).to(dtype=z.dtype)


class ConditionedTransitionBlock(nn.Module):
    def __init__(self, c_a: int, c_s: int, n: int = 2, biasinit: float = -2.0) -> None:
        super().__init__()
        self.adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        self.linear_a1 = LinearNoBias(c_a, n * c_a, initializer="relu")
        self.linear_a2 = LinearNoBias(c_a, n * c_a, initializer="relu")
        self.linear_b = LinearNoBias(n * c_a, c_a, initializer="zeros")
        self.linear_s = BiasInitLinear(c_s, c_a, bias=True, biasinit=biasinit)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        a = self.adaln(a, s)
        b = F.silu(self.linear_a1(a)) * self.linear_a2(a)
        return torch.sigmoid(self.linear_s(s)) * self.linear_b(b)


class DiffusionConditioning(nn.Module):
    def __init__(
        self,
        *,
        sigma_data: float,
        c_z: int,
        c_s: int,
        c_s_inputs: int,
        c_noise_embedding: int = 128,
    ) -> None:
        super().__init__()
        self.sigma_data = float(sigma_data)
        self.relpe = RelativePositionEncoding(c_z=c_z)
        self.layernorm_z = LayerNorm(2 * c_z, create_offset=False)
        self.linear_no_bias_z = LinearNoBias(2 * c_z, c_z)
        self.transition_z1 = Transition(c_z, n=2)
        self.transition_z2 = Transition(c_z, n=2)
        self.layernorm_s = LayerNorm(c_s + c_s_inputs, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(c_s + c_s_inputs, c_s)
        self.fourier_embedding = FourierEmbedding(c_noise_embedding)
        self.layernorm_n = LayerNorm(c_noise_embedding, create_offset=False)
        self.linear_no_bias_n = LinearNoBias(c_noise_embedding, c_s)
        self.transition_s1 = Transition(c_s, n=2)
        self.transition_s2 = Transition(c_s, n=2)

    def prepare_cache(
        self,
        *,
        relp_feature: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        pair_z = torch.cat([z_trunk, self.relpe(relp_feature)], dim=-1)
        pair_z = self.linear_no_bias_z(self.layernorm_z(pair_z))
        pair_z = pair_z + self.transition_z1(pair_z)
        pair_z = pair_z + self.transition_z2(pair_z)
        return pair_z * pair_mask.unsqueeze(-1).to(dtype=pair_z.dtype)

    def forward(
        self,
        *,
        noise_level: torch.Tensor,
        relp_feature: torch.Tensor,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_mask: torch.Tensor,
        mask: torch.Tensor,
        pair_z: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_z is None:
            pair_z = self.prepare_cache(relp_feature=relp_feature, z_trunk=z_trunk, pair_mask=pair_mask)

        single_s = torch.cat([s_trunk, s_inputs], dim=-1)
        single_s = self.linear_no_bias_s(self.layernorm_s(single_s))
        noise_embed = self.fourier_embedding(torch.log(noise_level.clamp(min=1e-4) / self.sigma_data) / 4.0)
        single_s = single_s + self.linear_no_bias_n(self.layernorm_n(noise_embed)).unsqueeze(1)
        single_s = single_s + self.transition_s1(single_s)
        single_s = single_s + self.transition_s2(single_s)
        single_s = single_s * _mask_f(mask, dtype=single_s.dtype)
        return single_s, pair_z


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, *, c_a: int, c_s: int, c_z: int, n_heads: int) -> None:
        super().__init__()
        self.attention_pair_bias = AttentionPairBias(
            has_s=True,
            n_heads=n_heads,
            c_a=c_a,
            c_s=c_s,
            c_z=c_z,
        )
        self.conditioned_transition_block = ConditionedTransitionBlock(c_a=c_a, c_s=c_s, n=2)

    def forward(self, a: torch.Tensor, s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a = a + self.attention_pair_bias(a=a, s=s, z=z, mask=mask)
        a = a + self.conditioned_transition_block(a, s)
        return a * _mask_f(mask, dtype=a.dtype)


class DiffusionTransformer(nn.Module):
    def __init__(self, *, c_a: int, c_s: int, c_z: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(c_a=c_a, c_s=c_s, c_z=c_z, n_heads=n_heads)
                for _ in range(max(1, int(n_blocks)))
            ]
        )

    def forward(self, a: torch.Tensor, s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            a = block(a=a, s=s, z=z, mask=mask)
        return a


class DiffusionModule(nn.Module):
    def __init__(
        self,
        *,
        sigma_data: float,
        c_a: int,
        c_s: int,
        c_z: int,
        c_s_inputs: int,
        n_blocks: int,
        n_heads: int,
        coord_step_size: float,
    ) -> None:
        super().__init__()
        self.coord_step_size = float(coord_step_size)
        self.diffusion_conditioning = DiffusionConditioning(
            sigma_data=sigma_data,
            c_z=c_z,
            c_s=c_s,
            c_s_inputs=c_s_inputs,
            c_noise_embedding=max(128, c_s),
        )
        self.coord_proj = nn.Linear(3, c_a)
        self.layernorm_s = LayerNorm(c_s, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(c_s, c_a, initializer="zeros")
        self.diffusion_transformer = DiffusionTransformer(
            c_a=c_a,
            c_s=c_s,
            c_z=c_z,
            n_blocks=n_blocks,
            n_heads=n_heads,
        )
        self.layernorm_a = LayerNorm(c_a, create_offset=False)
        self.output_proj = nn.Linear(c_a, 3)

    def forward(
        self,
        *,
        coords: torch.Tensor,
        noise_level: torch.Tensor,
        relp_feature: torch.Tensor,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_mask: torch.Tensor,
        mask: torch.Tensor,
        pair_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        s_single, z_pair = self.diffusion_conditioning(
            noise_level=noise_level,
            relp_feature=relp_feature,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_mask=pair_mask,
            mask=mask,
            pair_z=pair_z,
        )
        a = self.coord_proj(coords) + self.linear_no_bias_s(self.layernorm_s(s_single))
        a = a * _mask_f(mask, dtype=a.dtype)
        a = self.diffusion_transformer(a=a, s=s_single, z=z_pair, mask=mask)
        delta = self.output_proj(self.layernorm_a(a))
        return self.coord_step_size * delta * _mask_f(mask, dtype=delta.dtype)


class ProtenixStyleNet(nn.Module):
    def __init__(
        self,
        *,
        residue_vocab_size: int = 5,
        max_chain_embeddings: int = 64,
        max_copy_embeddings: int = 64,
        c_s: Optional[int] = None,
        c_z: int = 64,
        c_token: Optional[int] = None,
        c_m: int = 32,
        template_c: int = 32,
        diffusion_c_a: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        input_embedder_layers: int = 1,
        template_blocks: int = 2,
        msa_blocks: int = 1,
        pair_low_rank_dim: Optional[int] = None,
        gru_blocks: int = 4,
        pairformer_blocks: int = 8,
        diffusion_blocks: int = 8,
        diffusion_steps: int = 2,
        num_structure_candidates: int = 5,
        coord_step_size: float = 0.1,
        triangle_attention_mode: str = "linear",
        triangle_feature_map_multiplier: int = 2,
        use_chunked_triangle_multiplication: bool = True,
        triangle_multiplication_chunk_size: int = 64,
        diffusion_sampler: str = "ode",
        diffusion_eta: float = 1.0,
        diffusion_lambda: float = 1.0,
        diffusion_gamma0: float = 0.0,
        use_templates: bool = True,
        use_rna_msa: bool = True,
    ) -> None:
        super().__init__()
        del input_embedder_layers
        del pair_low_rank_dim
        del diffusion_lambda
        del diffusion_gamma0

        if hidden_dim is not None:
            if c_s is None:
                c_s = int(hidden_dim)
            if c_token is None:
                c_token = int(hidden_dim)

        self.c_s = int(192 if c_s is None else c_s)
        self.c_token = int(self.c_s if c_token is None else c_token)
        self.c_z = max(1, int(c_z))
        self.c_m = max(1, int(c_m))
        self.template_c = max(1, int(template_c))
        self.diffusion_c_a = max(1, int(self.c_s * 2 if diffusion_c_a is None else diffusion_c_a))
        self.diffusion_steps = max(1, int(diffusion_steps))
        self.num_structure_candidates = max(1, int(num_structure_candidates))
        self.gru_blocks = max(0, int(gru_blocks))
        self.use_templates = bool(use_templates)
        self.use_rna_msa = bool(use_rna_msa)
        self.diffusion_eta = float(diffusion_eta)
        self.residue_vocab_size = int(residue_vocab_size)
        self.triangle_attention_mode = str(triangle_attention_mode).strip().lower().replace("-", "_")
        self.triangle_feature_map_multiplier = max(1, int(triangle_feature_map_multiplier))
        self.use_chunked_triangle_multiplication = bool(use_chunked_triangle_multiplication)
        self.triangle_multiplication_chunk_size = max(1, int(triangle_multiplication_chunk_size))

        sampler_name = str(diffusion_sampler).strip().lower().replace("-", "_")
        sampler_aliases = {
            "ode": "ode",
            "ode2": "ode",
            "ode_2_step": "ode",
            "ode_2step": "ode",
        }
        if sampler_name not in sampler_aliases:
            raise ValueError(
                f"Unsupported diffusion_sampler='{diffusion_sampler}'. Expected one of: {sorted(sampler_aliases)}"
            )
        self.diffusion_sampler = sampler_aliases[sampler_name]

        self.input_embedder = InputFeatureEmbedder(
            residue_vocab_size=self.residue_vocab_size,
            max_chain_embeddings=max_chain_embeddings,
            max_copy_embeddings=max_copy_embeddings,
            c_token=self.c_token,
        )
        self.c_s_inputs = self.input_embedder.c_s_inputs

        self.linear_no_bias_sinit = LinearNoBias(self.c_s_inputs, self.c_s)
        self.linear_no_bias_zinit1 = LinearNoBias(self.c_s, self.c_z)
        self.linear_no_bias_zinit2 = LinearNoBias(self.c_s, self.c_z)
        self.relative_position_encoding = RelativePositionEncoding(c_z=self.c_z)

        self.template_embedder = TemplateEmbedder(
            n_blocks=template_blocks,
            c=self.template_c,
            c_z=self.c_z,
            residue_vocab_size=self.residue_vocab_size,
            n_heads=max(1, num_heads // 2),
            dropout=dropout,
        )
        self.msa_module = MSAModule(
            n_blocks=msa_blocks,
            c_m=self.c_m,
            c_z=self.c_z,
            c_s_inputs=self.c_s_inputs,
            residue_vocab_size=self.residue_vocab_size,
            msa_dropout=dropout,
            pair_dropout=dropout,
        )

        gru_hidden_dim = max(1, self.c_s // 2)
        if self.gru_blocks > 0:
            self.gru_norm: nn.Module = LayerNorm(self.c_s)
            self.frontend_gru: Optional[nn.GRU] = nn.GRU(
                input_size=self.c_s,
                hidden_size=gru_hidden_dim,
                num_layers=self.gru_blocks,
                dropout=dropout if self.gru_blocks > 1 else 0.0,
                bidirectional=True,
                batch_first=True,
            )
            self.gru_out_proj: Optional[nn.Module] = (
                nn.Identity()
                if gru_hidden_dim * 2 == self.c_s
                else nn.Linear(gru_hidden_dim * 2, self.c_s)
            )
            self.gru_dropout: nn.Module = nn.Dropout(dropout)
        else:
            self.gru_norm = nn.Identity()
            self.frontend_gru = None
            self.gru_out_proj = None
            self.gru_dropout = nn.Identity()

        self.pairformer_stack = PairformerStack(
            n_blocks=pairformer_blocks,
            c_z=self.c_z,
            c_s=self.c_s,
            n_heads=num_heads,
            dropout=dropout,
            triangle_attention_mode=self.triangle_attention_mode,
            triangle_feature_map_multiplier=self.triangle_feature_map_multiplier,
            use_chunked_triangle_multiplication=self.use_chunked_triangle_multiplication,
        )
        self.diffusion_module = DiffusionModule(
            sigma_data=16.0,
            c_a=self.diffusion_c_a,
            c_s=self.c_s,
            c_z=self.c_z,
            c_s_inputs=self.c_s_inputs,
            n_blocks=diffusion_blocks,
            n_heads=num_heads,
            coord_step_size=coord_step_size,
        )
        self.structure_candidate_embed = nn.Embedding(self.num_structure_candidates, self.c_s)

    def _masked_coord_center(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = _mask_f(mask, dtype=coords.dtype)
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (coords * mask_f).sum(dim=1, keepdim=True) / denom

    def _prepare_optional_bpp(
        self,
        coords: torch.Tensor,
        rna_bpp_banded: Optional[torch.Tensor],
        rna_bpp_mask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        bpp = None if rna_bpp_banded is None else rna_bpp_banded.to(device=coords.device, dtype=coords.dtype)
        bpp_mask_t = None if rna_bpp_mask is None else rna_bpp_mask.to(device=coords.device, dtype=torch.bool)
        return bpp, bpp_mask_t

    def _build_msa_profile(
        self,
        rna_msa_tokens: torch.Tensor,
        rna_msa_mask: torch.Tensor,
        rna_msa_row_valid: Optional[torch.Tensor],
        coords_dtype: torch.dtype,
    ) -> torch.Tensor:
        row_valid = (
            torch.ones(rna_msa_tokens.shape[:2], device=rna_msa_tokens.device, dtype=torch.bool)
            if rna_msa_row_valid is None
            else rna_msa_row_valid.to(device=rna_msa_tokens.device, dtype=torch.bool)
        )
        token_mask = rna_msa_mask.to(dtype=torch.bool) & row_valid.unsqueeze(-1)
        one_hot = F.one_hot(
            rna_msa_tokens.clamp(min=0, max=self.residue_vocab_size - 1),
            num_classes=self.residue_vocab_size,
        ).to(dtype=coords_dtype)
        numer = (one_hot * token_mask.unsqueeze(-1).to(dtype=coords_dtype)).sum(dim=1)
        denom = token_mask.sum(dim=1, keepdim=False).unsqueeze(-1).clamp(min=1).to(dtype=coords_dtype)
        return numer / denom

    def _prepare_template_inputs(
        self,
        *,
        residue_idx: torch.Tensor,
        mask: torch.Tensor,
        template_coords: Optional[torch.Tensor],
        template_mask: Optional[torch.Tensor],
        template_topk_coords: Optional[torch.Tensor],
        template_topk_mask: Optional[torch.Tensor],
        template_topk_valid: Optional[torch.Tensor],
        template_topk_residue_idx: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if template_topk_coords is None and template_coords is not None:
            template_topk_coords = template_coords.unsqueeze(1)
            template_topk_mask = (
                mask.unsqueeze(1) if template_mask is None else template_mask.unsqueeze(1)
            )
            template_topk_valid = template_topk_mask.any(dim=-1)
            template_topk_residue_idx = residue_idx.unsqueeze(1)
        if template_topk_coords is None:
            return None, None, None, None

        if template_topk_mask is None:
            template_topk_mask = torch.ones(
                template_topk_coords.shape[:-1],
                device=template_topk_coords.device,
                dtype=torch.bool,
            )
        if template_topk_valid is None:
            template_topk_valid = template_topk_mask.any(dim=-1)
        if template_topk_residue_idx is None:
            template_topk_residue_idx = residue_idx.unsqueeze(1).expand(-1, template_topk_coords.shape[1], -1)
        return template_topk_coords, template_topk_mask, template_topk_valid, template_topk_residue_idx

    def _prepare_structure_candidates(
        self,
        *,
        coords_centered: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_coords = coords_centered.unsqueeze(1).expand(-1, self.num_structure_candidates, -1, -1).contiguous()
        candidate_mask = mask.unsqueeze(1).expand(-1, self.num_structure_candidates, -1)
        candidate_valid = torch.ones(
            coords_centered.shape[0],
            self.num_structure_candidates,
            device=coords_centered.device,
            dtype=torch.bool,
        )
        return candidate_coords, candidate_mask, candidate_valid

    def _run_structure_candidate_diffusion(
        self,
        *,
        candidate_coords: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_valid: torch.Tensor,
        coord_center: torch.Tensor,
        coords_dtype: torch.dtype,
        mask: torch.Tensor,
        relp_feature: torch.Tensor,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_mask: torch.Tensor,
        pair_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n_candidates, seq_len, _ = candidate_coords.shape
        flat_mask = mask.unsqueeze(1).expand(-1, n_candidates, -1).reshape(bsz * n_candidates, seq_len)
        flat_mask_f = _mask_f(flat_mask, dtype=coords_dtype)
        flat_coords = candidate_coords.reshape(bsz * n_candidates, seq_len, 3)
        flat_center = self._masked_coord_center(flat_coords, flat_mask)
        current_coords = (flat_coords - flat_center) * flat_mask_f

        def _repeat_across_candidates(x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(1).expand(-1, n_candidates, *x.shape[1:]).reshape(bsz * n_candidates, *x.shape[1:])

        flat_relp_feature = _repeat_across_candidates(relp_feature)
        flat_s_inputs = _repeat_across_candidates(s_inputs)
        flat_s_trunk = _repeat_across_candidates(s_trunk)
        flat_z_trunk = _repeat_across_candidates(z_trunk)
        flat_pair_mask = _repeat_across_candidates(pair_mask)
        flat_pair_z = _repeat_across_candidates(pair_z)
        slot_bias = self.structure_candidate_embed.weight.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * n_candidates, self.c_s)
        flat_s_trunk = flat_s_trunk + slot_bias.unsqueeze(1)

        noise_schedule = torch.linspace(1.0, 0.2, steps=self.diffusion_steps, device=flat_coords.device, dtype=coords_dtype)
        for noise_level in noise_schedule:
            sigma = current_coords.new_full((current_coords.shape[0],), float(noise_level))
            candidate_delta = self.diffusion_module(
                coords=current_coords,
                noise_level=sigma,
                relp_feature=flat_relp_feature,
                s_inputs=flat_s_inputs,
                s_trunk=flat_s_trunk,
                z_trunk=flat_z_trunk,
                pair_mask=flat_pair_mask,
                mask=flat_mask,
                pair_z=flat_pair_z,
            )
            current_coords = (current_coords + self._apply_diffusion_sampler(candidate_delta)) * flat_mask_f

        corrected_templates = (current_coords + flat_center).reshape(bsz, n_candidates, seq_len, 3)
        corrected_templates = corrected_templates * mask.unsqueeze(1).unsqueeze(-1).to(dtype=coords_dtype)
        corrected_templates = corrected_templates + coord_center.unsqueeze(1)

        valid_f = candidate_valid.to(dtype=coords_dtype).unsqueeze(-1).unsqueeze(-1)
        aggregated_coords = corrected_templates[:, 0]
        if n_candidates > 1:
            aggregated_coords = (corrected_templates * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1.0)
        aggregated_coords = aggregated_coords * _mask_f(mask, dtype=coords_dtype)
        corrected_mask = mask.unsqueeze(1).expand(bsz, n_candidates, seq_len)
        return aggregated_coords, corrected_templates, corrected_mask, candidate_valid

    def _run_gru_frontend(self, single: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.frontend_gru is None:
            return single

        lengths = mask.sum(dim=1).to(dtype=torch.long)
        packed = pack_padded_sequence(
            self.gru_norm(single),
            lengths=lengths.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.frontend_gru(packed)
        gru_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=single.shape[1])
        assert self.gru_out_proj is not None
        single = single + self.gru_dropout(self.gru_out_proj(gru_out))
        return single * _mask_f(mask, dtype=single.dtype)

    def _apply_diffusion_sampler(self, candidate_delta: torch.Tensor) -> torch.Tensor:
        if self.diffusion_sampler != "ode":
            raise RuntimeError(f"Unsupported diffusion sampler at runtime: {self.diffusion_sampler}")
        return candidate_delta * candidate_delta.new_tensor(self.diffusion_eta)

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
        rna_msa_tokens: Optional[torch.Tensor] = None,
        rna_msa_mask: Optional[torch.Tensor] = None,
        rna_msa_row_valid: Optional[torch.Tensor] = None,
        rna_msa_profile: Optional[torch.Tensor] = None,
        rna_bpp_banded: Optional[torch.Tensor] = None,
        rna_bpp_mask: Optional[torch.Tensor] = None,
        return_aux_outputs: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del template_topk_identity
        del template_topk_similarity

        residue_idx = residue_idx.clamp(min=0, max=self.residue_vocab_size - 1)
        chain_idx = chain_idx.clamp(min=0, max=self.input_embedder.chain_emb.num_embeddings - 1)
        copy_idx = copy_idx.clamp(min=0, max=self.input_embedder.copy_emb.num_embeddings - 1)

        mask_f = _mask_f(mask, dtype=coords.dtype)
        pair_mask = _pair_mask(mask)
        pair_mask_f = pair_mask.unsqueeze(-1).to(dtype=coords.dtype)

        coord_center = self._masked_coord_center(coords, mask)
        coords_centered = (coords - coord_center) * mask_f

        if rna_msa_profile is None and rna_msa_tokens is not None and rna_msa_mask is not None:
            rna_msa_profile = self._build_msa_profile(
                rna_msa_tokens=rna_msa_tokens,
                rna_msa_mask=rna_msa_mask,
                rna_msa_row_valid=rna_msa_row_valid,
                coords_dtype=coords.dtype,
            )

        rna_bpp_banded_t, rna_bpp_mask_t = self._prepare_optional_bpp(
            coords=coords_centered,
            rna_bpp_banded=rna_bpp_banded,
            rna_bpp_mask=rna_bpp_mask,
        )
        s_inputs = self.input_embedder(
            residue_idx=residue_idx,
            chain_idx=chain_idx,
            copy_idx=copy_idx,
            resid=resid,
            coords=coords_centered,
            mask=mask,
            rna_msa_profile=rna_msa_profile,
            rna_bpp_banded=rna_bpp_banded_t,
            rna_bpp_mask=rna_bpp_mask_t,
        )

        s_init = self.linear_no_bias_sinit(s_inputs) * mask_f
        relp_feature = self.relative_position_encoding.build_feature(chain_idx=chain_idx, copy_idx=copy_idx, mask=mask)
        z = (
            self.linear_no_bias_zinit1(s_init).unsqueeze(2)
            + self.linear_no_bias_zinit2(s_init).unsqueeze(1)
            + self.relative_position_encoding(relp_feature)
        )
        z = 0.5 * (z + z.transpose(1, 2))
        z = z * pair_mask_f

        template_topk_coords, template_topk_mask, template_topk_valid, template_topk_residue_idx = self._prepare_template_inputs(
            residue_idx=residue_idx,
            mask=mask,
            template_coords=template_coords,
            template_mask=template_mask,
            template_topk_coords=template_topk_coords,
            template_topk_mask=template_topk_mask,
            template_topk_valid=template_topk_valid,
            template_topk_residue_idx=template_topk_residue_idx,
        )
        if self.use_templates:
            z = z + self.template_embedder(
                z=z,
                pair_mask=pair_mask,
                template_topk_coords=template_topk_coords,
                template_topk_mask=template_topk_mask,
                template_topk_valid=template_topk_valid,
                template_topk_residue_idx=template_topk_residue_idx,
            )
            z = 0.5 * (z + z.transpose(1, 2))
            z = z * pair_mask_f

        if self.use_rna_msa:
            if rna_msa_tokens is None or rna_msa_mask is None:
                raise RuntimeError("ProtenixStyleNet requires RNA MSA tensors when `use_rna_msa=True`.")
            z = self.msa_module(
                rna_msa_tokens=rna_msa_tokens,
                rna_msa_mask=rna_msa_mask,
                rna_msa_row_valid=rna_msa_row_valid,
                s_inputs=s_inputs,
                z=z,
                pair_mask=pair_mask,
            )

        s = self._run_gru_frontend(s_init, mask)
        chunk_ids = (
            _build_chain_chunk_ids(chain_idx=chain_idx, mask=mask, target_chunk_size=self.triangle_multiplication_chunk_size)
            if self.use_chunked_triangle_multiplication
            else None
        )
        s, z = self.pairformer_stack(s=s, z=z, pair_mask=pair_mask, mask=mask, chunk_ids=chunk_ids)
        assert s is not None

        pair_z = self.diffusion_module.diffusion_conditioning.prepare_cache(
            relp_feature=relp_feature,
            z_trunk=z,
            pair_mask=pair_mask,
        )
        candidate_coords, candidate_mask, candidate_valid = self._prepare_structure_candidates(
            coords_centered=coords_centered,
            mask=mask,
        )
        aggregated_coords, corrected_templates, corrected_mask, corrected_valid = self._run_structure_candidate_diffusion(
            candidate_coords=candidate_coords,
            candidate_mask=candidate_mask,
            candidate_valid=candidate_valid,
            coord_center=coord_center,
            coords_dtype=coords.dtype,
            mask=mask,
            relp_feature=relp_feature,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=pair_mask,
            pair_z=pair_z,
        )

        if not return_aux_outputs:
            return aggregated_coords
        return {
            "coords": aggregated_coords,
            "candidate_topk_coords": corrected_templates,
            "candidate_topk_mask": corrected_mask,
            "candidate_topk_valid": corrected_valid,
        }


if __name__ == "__main__":
    _ = ProtenixStyleNet()
