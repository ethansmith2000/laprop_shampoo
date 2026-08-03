import math
from typing import Optional
import torch


def _xavier_fused(weight: torch.Tensor, n_chunks: int, dim: int = 0):
    """Xavier-init a fused projection as if it were `n_chunks` separate ones.

    `xavier_uniform_` derives fan_out from the tensor shape, so on a fused
    [n*out, in] weight it sees fan_out = n*out and produces
    std = sqrt(2/(in + n*out)) where n separate [out, in] projections would
    each get sqrt(2/(in + out)). The fused weight comes out too small by
    sqrt((in+out)/(in+n*out)) -- 0.71x for a [3d, d] QKV, 0.76x for the
    [2*hidden, d] GeGLU value+gate. Initializing each chunk independently
    restores parity with the unfused parameterization.
    """
    for chunk in weight.chunk(n_chunks, dim=dim):
        torch.nn.init.xavier_uniform_(chunk)




class Attention(torch.nn.Module):
    def __init__(
        self,
        dim,
        heads,
        is_causal=True,
        use_rope=True,
        rope_theta=10000.0,
        qk_norm=True,
        max_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.is_causal = is_causal
        self.use_rope = use_rope
        self.qk_norm = qk_norm
        self.head_dim = dim // heads
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len

        if self.use_rope and (self.head_dim % 2 != 0):
            raise ValueError("RoPE requires head_dim to be even.")

        # Fused QKV projection: one matmul instead of three.
        self.to_qkv = torch.nn.Linear(dim, 3 * dim, bias=False)
        self.to_out = torch.nn.Linear(dim, dim, bias=True)

        self.q_norm = torch.nn.RMSNorm(self.head_dim) if qk_norm else torch.nn.Identity()
        self.k_norm = torch.nn.RMSNorm(self.head_dim) if qk_norm else torch.nn.Identity()

        # Lazily filled / grown RoPE cache (persistent=False so checkpoints stay lean).
        self.register_buffer("_rope_cos", None, persistent=False)
        self.register_buffer("_rope_sin", None, persistent=False)

    def _rope_tables(self, seq_len: int, device: torch.device):
        """Return cached cos/sin tables in fp32, allocating or growing as needed.

        The tables are always built and cached in fp32, never in the autocast
        dtype. bf16 has 8 mantissa bits, so integer positions above 256 are not
        exactly representable (1023 rounds to 1024) and the angle -- whose
        magnitude runs to ~seq_len radians -- loses all its low bits before the
        trig call. Measured at seq_len=1024 that is up to 1.97 absolute error in
        cos, i.e. a sign flip. Callers cast the result to the working dtype,
        where the values lie in [-1, 1] and bf16 precision is adequate.
        """
        need_build = (
            self._rope_cos is None
            or self._rope_cos.device != device
            or self._rope_cos.shape[-2] < seq_len
        )
        if need_build:
            # Prefer a stable upper bound when known (avoids rebuilds as T varies).
            build_len = seq_len
            if self.max_seq_len is not None:
                build_len = max(seq_len, self.max_seq_len)
            half = self.head_dim // 2
            # autocast must be off for the build: matmul-family ops (einsum,
            # outer) are on the bf16 cast list and would silently downcast the
            # fp32 inputs, reintroducing exactly the precision loss this avoids.
            with torch.amp.autocast(device_type=device.type, enabled=False):
                # Accumulate the angle in fp64 and store the trig result in
                # fp32. The angle reaches ~build_len radians, so fp32 rounding
                # there costs ~1e-4 absolute at 1k context and grows linearly
                # with it; doing the one-time build wide keeps the stored
                # tables exact to fp32 at any context length.
                freqs = torch.arange(half, device=device, dtype=torch.float64)
                inv_freq = 1.0 / (self.rope_theta ** (freqs / half))
                positions = torch.arange(
                    build_len, device=device, dtype=torch.float64
                )
                angles = positions[:, None] * inv_freq[None, :]
                # Shape: [1, 1, T, half] for broadcast over batch/heads.
                self._rope_cos = angles.cos()[None, None, :, :].float()
                self._rope_sin = angles.sin()[None, None, :, :].float()
        return self._rope_cos[..., :seq_len, :], self._rope_sin[..., :seq_len, :]

    def _apply_rope(self, q, k):
        if not self.use_rope:
            return q, k

        _, _, t, d = q.shape
        half = d // 2
        cos, sin = self._rope_tables(t, q.device)
        # Cast at use: the tables stay fp32 in the cache, the rotation runs in
        # the activation dtype.
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)

        def rotate(x):
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return rotate(q), rotate(k)

    def forward(self, x):
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2), (q, k, v))
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self._apply_rope(q, k)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        # Bring sequence length back to axis 1 before merging heads
        attn = attn.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        return self.to_out(attn)


class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class GeGLU(torch.nn.Module):
    def __init__(self, dim, hidden_dim=None, align_multiple=64):
        super().__init__()
        # Default to GeGLU parity (~2.67x) and align to a friendly GPU multiple
        # if hidden_dim is None:
        hidden_dim = math.ceil(dim * 8 / 3)
        if align_multiple is not None and align_multiple > 1:
            hidden_dim = math.ceil(hidden_dim / align_multiple) * align_multiple
        # Fused value+gate projection: one matmul, then chunk.
        self.proj_in = torch.nn.Linear(dim, 2 * hidden_dim)
        self.proj_out = torch.nn.Linear(hidden_dim, dim)
        self.act = torch.nn.GELU()

    def forward(self, x):
        value, gate = self.proj_in(x).chunk(2, dim=-1)
        return self.proj_out(value * self.act(gate))

class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        dim,
        heads,
        ff_hidden_dim,
        permute_before_attn=False,
        permute_before_mlp=False,
        permute_kwargs=None,
        is_causal=True,
        use_rope=True,
        rope_theta=10000.0,
        qk_norm=True,
        max_seq_len: Optional[int] = None,
    ):
        super().__init__()
        permute_kwargs = permute_kwargs or {}

        self.attn = Attention(
            dim,
            heads,
            is_causal=is_causal,
            use_rope=use_rope,
            rope_theta=rope_theta,
            qk_norm=qk_norm,
            max_seq_len=max_seq_len,
        )
        self.ff = GeGLU(dim, hidden_dim=ff_hidden_dim)
        self.norm1 = torch.nn.RMSNorm(dim)
        self.norm2 = torch.nn.RMSNorm(dim)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        ff_mult,
        vocab_size,
        max_seq_len,
        gradient_checkpointing=False,
        use_rope=True,
        rope_theta=10000.0,
        qk_norm=True,
        depth_scaled_residual_init=True,
    ):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, dim)
        self.use_rope = use_rope
        if not use_rope:
            self.position_embedding = torch.nn.Embedding(max_seq_len, dim)
        self.in_proj = torch.nn.Sequential(
            torch.nn.RMSNorm(dim),
            torch.nn.Linear(dim, dim),
        )
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(
                dim,
                heads,
                dim * ff_mult,
                is_causal=True,
                use_rope=use_rope,
                rope_theta=rope_theta,
                qk_norm=qk_norm,
                max_seq_len=max_seq_len,
            ) for _ in range(depth)])
        self.out_proj = torch.nn.Sequential(
            torch.nn.RMSNorm(dim),
            torch.nn.Linear(dim, vocab_size),
        )
        self.gradient_checkpointing = gradient_checkpointing

        # Initialize weights following common transformer best practices
        embed_std = dim ** -0.5
        lm_head_std = 0.02

        with torch.no_grad():
            for name, module in self.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=embed_std)
                elif isinstance(module, torch.nn.LayerNorm):
                    torch.nn.init.ones_(module.weight)
                    torch.nn.init.zeros_(module.bias)
                elif isinstance(module, torch.nn.RMSNorm):
                    torch.nn.init.ones_(module.weight)
                elif isinstance(module, torch.nn.Linear):
                    # Skip LoRA permuter submodules that already perform their own init
                    if "permute." in name:
                        continue
                    # Fused projections must be initialized chunk-wise; a single
                    # xavier call on the fused weight uses the wrong fan_out.
                    if "attn.to_qkv" in name:
                        _xavier_fused(module.weight, 3)
                    elif "ff.proj_in" in name:
                        _xavier_fused(module.weight, 2)   # value + gate
                    else:
                        torch.nn.init.xavier_uniform_(module.weight)

                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

            torch.nn.init.normal_(self.out_proj[1].weight, mean=0.0, std=lm_head_std)
            torch.nn.init.zeros_(self.out_proj[1].bias)

            # Damp the two projections that write into the residual stream so
            # its variance stays ~flat with depth instead of compounding. Kept
            # as a flag because it moves the baseline: it is not a bug fix.
            self.depth_scaled_residual_init = depth_scaled_residual_init
            if depth_scaled_residual_init:
                residual_scale = (2.0 * depth) ** -0.5
                for block in self.blocks:
                    block.attn.to_out.weight.mul_(residual_scale)
                    block.ff.proj_out.weight.mul_(residual_scale)

    def forward(self, input_ids, targets=None, return_logits=False):
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device)
        tok_emb = self.token_embedding(input_ids)
        if not self.use_rope:
            pos_emb = self.position_embedding(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb
        x = self.in_proj(x)
        if self.gradient_checkpointing:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, preserve_rng_state=False, use_reentrant=False, determinism_check="none")
        else:
            for block in self.blocks:
                x = block(x)
        if targets is not None:
            logits = self.out_proj(x)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, -2), targets.reshape(-1))
            if return_logits:
                return loss, logits
            return loss
        return self.out_proj(x)

    def resize_token_embeddings(self, new_size: int):
        if not isinstance(self.token_embedding, torch.nn.Embedding):
            raise NotImplementedError("resize_token_embeddings only supports dense Embedding tables.")

        old_weight = self.token_embedding.weight
        new_emb = torch.nn.Embedding(
            new_size,
            old_weight.shape[1],
            device=old_weight.device,
            dtype=old_weight.dtype,
        )
        with torch.no_grad():
            num_tokens = min(old_weight.shape[0], new_size)
            new_emb.weight[:num_tokens] = old_weight[:num_tokens]
        self.token_embedding = new_emb
        return self.token_embedding
