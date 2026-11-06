"""A configurable GPT.

Every component this project ablates is selectable from `ModelConfig`, so an
experiment changes a config field and never the code. That matters for honesty
as much as convenience: two runs differ only in the field under test, so a
measured difference is attributable to it.

Implemented variants:

    normalisation   LayerNorm        RMSNorm
    activation      GELU  ReLU  SiLU  SwiGLU
    position        learned          RoPE            none (control)
    attention       multi-head       grouped-query
    masking         causal           unmasked (control)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# ------------------------------------------------------------ normalisation --


class LayerNorm(nn.Module):
    """Centres and rescales. Written out rather than using nn.LayerNorm so the
    contrast with RMSNorm is visible: this subtracts the mean, RMSNorm does not."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class RMSNorm(nn.Module):
    """Rescales by root-mean-square only - no mean subtraction, no shift.

    Fewer operations and one fewer parameter tensor per norm; used by Llama and
    most recent architectures on the finding that re-centring contributes little.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * x * rms


def make_norm(kind: str, dim: int) -> nn.Module:
    return {"layernorm": LayerNorm, "rmsnorm": RMSNorm}[kind](dim)


# --------------------------------------------------------------- activation --


class GELU(nn.Module):
    """Exact-ish GELU via the tanh approximation, as used by GPT-2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Position-wise MLP.

    SwiGLU is shaped differently from the others: it splits the projection into
    a value path and a gate path, so to keep the parameter count comparable the
    hidden dimension is scaled by 2/3 (the convention Llama uses).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.kind = cfg.activation
        hidden = 4 * cfg.emb_dim

        if self.kind == "swiglu":
            hidden = int(2 * hidden / 3)
            self.up = nn.Linear(cfg.emb_dim, hidden, bias=False)
            self.gate = nn.Linear(cfg.emb_dim, hidden, bias=False)
            self.down = nn.Linear(hidden, cfg.emb_dim, bias=False)
        else:
            self.up = nn.Linear(cfg.emb_dim, hidden)
            self.down = nn.Linear(hidden, cfg.emb_dim)
            self.act = {"gelu": GELU(), "relu": nn.ReLU(), "silu": nn.SiLU()}[self.kind]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "swiglu":
            return self.down(F.silu(self.gate(x)) * self.up(x))
        return self.down(self.act(self.up(x)))


# ----------------------------------------------------------------- position --


def build_rope_cache(head_dim: int, max_len: int, base: float = 10_000.0):
    """Precompute rotary sin/cos tables.

    RoPE encodes position by *rotating* query and key vectors by an angle
    proportional to their position, so the attention dot product depends on the
    difference between positions. Position information therefore enters
    attention directly and needs no learned embedding table.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_len).float()
    angles = torch.outer(positions, inv_freq)          # (max_len, head_dim/2)
    angles = torch.cat([angles, angles], dim=-1)       # (max_len, head_dim)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (batch, heads, seq, head_dim)
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


# ---------------------------------------------------------------- attention --


class Attention(nn.Module):
    """Multi-head or grouped-query attention, optionally rotary, optionally
    without a causal mask."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.kv_heads
        self.head_dim = cfg.head_dim

        self.q_proj = nn.Linear(cfg.emb_dim, cfg.n_heads * self.head_dim, bias=cfg.qkv_bias)
        self.k_proj = nn.Linear(cfg.emb_dim, self.n_kv * self.head_dim, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(cfg.emb_dim, self.n_kv * self.head_dim, bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.drop_rate)

        if cfg.causal_mask:
            mask = torch.triu(torch.ones(cfg.context_length, cfg.context_length), diagonal=1).bool()
            self.register_buffer("mask", mask, persistent=False)
        else:
            self.mask = None

        if cfg.position == "rope":
            cos, sin = build_rope_cache(self.head_dim, cfg.context_length)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)

        if self.cfg.position == "rope":
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Grouped-query attention: each key/value head serves several query
        # heads, cutting KV memory without changing the query count.
        if self.n_kv != self.n_heads:
            repeat = self.n_heads // self.n_kv
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.mask is not None:
            scores = scores.masked_fill(self.mask[:t, :t], float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))

        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, -1)
        return self.out_proj(out)


# -------------------------------------------------------------------- block --


class TransformerBlock(nn.Module):
    """Pre-norm block: normalise, sublayer, residual add.

    Pre-norm (rather than the original post-norm) is what allows deep stacks to
    train without a learning-rate warmup crutch, because the residual path stays
    unnormalised end to end.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = make_norm(cfg.norm, cfg.emb_dim)
        self.attn = Attention(cfg)
        self.norm2 = make_norm(cfg.norm, cfg.emb_dim)
        self.ff = FeedForward(cfg)
        self.drop = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# -------------------------------------------------------------------- model --


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        if cfg.position == "learned":
            self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        else:
            # "rope" injects position inside attention; "none" is the control
            # that removes positional information entirely.
            self.pos_emb = None

        self.drop = nn.Dropout(cfg.drop_rate)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = make_norm(cfg.norm, cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.out_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(t, device=idx.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.out_head(self.final_norm(x))

    # ----------------------------------------------------------- accounting --

    def parameter_counts(self) -> dict[str, int]:
        """Split parameters into embedding vs transformer.

        Reported separately because at this scale the embedding table dominates,
        and "20M parameters" means something very different depending on how it
        splits.
        """
        total = sum(p.numel() for p in self.parameters())
        embedding = self.tok_emb.weight.numel()
        if self.pos_emb is not None:
            embedding += self.pos_emb.weight.numel()
        head = 0 if self.cfg.tie_embeddings else self.out_head.weight.numel()
        return {
            "total": total,
            "embedding": embedding,
            "output_head": head,
            "transformer": total - embedding - head,
        }

    # ---------------------------------------------------------- generation --

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            window = idx[:, -self.cfg.context_length :]
            logits = self(window)[:, -1, :]

            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            if temperature <= 0.0:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx
