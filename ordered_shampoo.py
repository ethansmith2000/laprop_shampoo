"""
OrderedShampoo — a unified optimizer for studying the interaction between
*whitening order* and *momentum* in matrix-preconditioned methods.

This implements the 2x3 family described in the research brief:

    mode in {'shampoo', 'eigenbasis'}  x  order in {'post', 'pre', 'pre_stale'}

`order` controls where momentum sits relative to the preconditioner:

    'post'       m <- EMA(g);  U = P(m)
                 Standard ordering. Shampoo / Adam / SOAP live here.

    'pre'        U = EMA(P(g))
                 LaProp-style. Momentum integrates in whitened coordinates,
                 so a preconditioner refresh rotates the buffer once rather
                 than applying a new geometry to a buffer accumulated under
                 the old one.

    'pre_stale'  U = EMA(clip(P_{t-1}(g_t)))   [stats updated AFTER the step]
                 ADOPT-style. The current gradient is excluded from its own
                 preconditioner, which is what removes the bias term behind
                 Adam's non-convergence. Includes ADOPT's c_t = t^(1/4)
                 clipping schedule for cold-start preconditioners.

`mode` controls the preconditioner:

    'shampoo'    Full Kronecker-factored inverse root: G <- x_i M_i^(-1/(2k)).
                 The whitening operator rotates, not just rescales, so 'pre'
                 and 'post' can point in genuinely different directions.

    'eigenbasis' SOAP-style: rotate into the eigenbasis of the Kronecker
                 factors, run a *diagonal* second moment there, rotate back.
                 Since SOAP is only a basis transform, applying LaProp/ADOPT
                 ordering inside the eigenbasis is a lightweight transfer of
                 the diagonal result -- this is the cheap fifth/sixth
                 condition in the experimental plan.

Tensors with ndim < 2 (or whose every dim exceeds max_precond_dim) fall back
to a diagonal path that, with the same `order` flag, is exactly
Adam / LaProp / ADOPT. So the same object covers baselines and proposals.

Optionally applies the corrected black-box Nesterov formula from
Kevin Yin's note (research.novelai.net/nesterov):

    phi_{t+1} = phi_t + (1 + mu_{t+1}) U_t - mu_t U_{t-1}

For Shampoo the previous update cannot be cheaply recomputed from the
buffers (the inverse root is the expensive part), so this costs one extra
parameter-sized state tensor. Off by default.

VALIDATION NOTE: the core tensor algebra (the dim-0-contract-append-to-end
cycling identity, the eigh-based inverse root, the eigenbasis roundtrip) and
the three ordering semantics were verified numerically against the Reddi et
al. counterexample in a NumPy mirror before this was written. In that test
'post' and 'pre' both diverge to the worst point while 'pre_stale' converges
-- i.e. reordering alone does NOT fix non-convergence, only the staleness
does. The PyTorch paths below have NOT been executed (no torch in the
authoring environment); run the self-test at the bottom first.
"""

import math
from typing import Callable, Iterable, List, Optional

import torch
from torch.optim import Optimizer


# --------------------------------------------------------------------------
# linear algebra helpers
# --------------------------------------------------------------------------

def _sym(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.transpose(-2, -1))


def _eigh_stable(M: torch.Tensor):
    """Symmetric eigendecomposition in float64, returned in M's dtype."""
    work = _sym(M).double()
    try:
        evals, evecs = torch.linalg.eigh(work)
    except Exception:
        # jitter and retry once; eigh occasionally fails on near-degenerate input
        n = work.shape[-1]
        jitter = 1e-10 * work.diagonal().abs().mean().clamp_min(1e-30)
        evals, evecs = torch.linalg.eigh(
            work + jitter * torch.eye(n, device=work.device, dtype=work.dtype)
        )
    return evals, evecs


def matrix_inverse_root(
    M: torch.Tensor, root: int, rel_eps: float = 1e-12
) -> torch.Tensor:
    """M^(-1/root) for symmetric PSD M, with relative eigenvalue flooring.

    Eigenvalues are clamped to rel_eps * lambda_max so the result stays
    bounded when M is rank deficient (common early in training).
    """
    evals, evecs = _eigh_stable(M)
    evals = evals.clamp_min(0)
    lmax = evals.max()
    if lmax <= 0:
        return torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
    evals = evals.clamp_min(lmax * rel_eps)
    inv = evals.pow(-1.0 / root)
    out = (evecs * inv.unsqueeze(-2)) @ evecs.transpose(-2, -1)
    return out.to(M.dtype)


def matrix_eigvecs(M: torch.Tensor) -> torch.Tensor:
    """Eigenvectors of symmetric M, columns ordered by ascending eigenvalue."""
    _, evecs = _eigh_stable(M)
    return evecs.to(M.dtype)


def _power_iteration_lmax(M: torch.Tensor, iters: int = 15) -> torch.Tensor:
    """Largest eigenvalue of symmetric PSD M, by power iteration.

    Used instead of the Frobenius norm to normalize before the NDB iteration.
    ||M||_F over-estimates the spectral norm badly for wide spectra, which
    scales the matrix too far down and stalls (then explodes) the iteration --
    DASH makes the same point, and we reproduced the failure.
    """
    v = torch.randn(M.shape[-1], 1, device=M.device, dtype=M.dtype)
    v = v / v.norm().clamp_min(1e-30)
    for _ in range(iters):
        v = M @ v
        v = v / v.norm().clamp_min(1e-30)
    return (v.transpose(-2, -1) @ (M @ v)).reshape(()).clamp_min(1e-30)


def _ndb_pass(A: torch.Tensor, iters: int):
    """One Newton-Denman-Beavers run: returns (A^(1/2), A^(-1/2)).

    E = (3I - ZY)/2 ;  Y <- YE ;  Z <- EZ.  Matmuls only, so this runs on
    tensor cores rather than cuSOLVER's sequential eigendecomposition.
    """
    n = A.shape[-1]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    Y, Z = A, I.clone()
    for _ in range(iters):
        E = 0.5 * (3 * I - Z @ Y)
        Y, Z = Y @ E, E @ Z
    return Y, Z


def matrix_inverse_root_ndb(
    M: torch.Tensor, root: int, damp: float = 1e-10, iters: int = 20
) -> torch.Tensor:
    """M^(-1/root) via repeated NDB inverse square roots. Power-of-two roots.

    Each pass halves the exponent, so root=4 (the 2-D Shampoo case) costs two
    passes. Runs in fp32: DASH reports NDB destabilizes in half precision, and
    we measured bf16 going to NaN. fp32 and fp64 were measured identical to two
    decimals of update angle at every damping level, so fp64 buys nothing here.

    DAMPING IS THE ACCURACY KNOB, not precision or iteration count. Measured
    update-direction error against an exact fp64 eigh, 60 iters, n=1024:

        damp   1e-6    1e-8    1e-10   1e-12
        err    5.5deg  0.09deg 0.00deg 0.00deg

    Hence the 1e-10 default. Judge approximations by update angle, never by
    relative error on the root itself: the root's error concentrates in
    low-eigenvalue directions carrying little gradient energy, so the two
    metrics disagree by orders of magnitude.

    CONDITIONING IS THE REAL LIMIT. On real factors at damp=1e-10, 20 iters,
    NDB is exact (<0.02 deg) wherever cond <= ~1e6 and 1.1-3x faster than fp64
    eigh. It degrades where the factors do not: attention to_qkv reaches cond
    ~1e10, where NDB gives 2-14 deg and needs so many iterations to recover
    that it loses to eigh outright. Prefer root_method='eigh' for those.
    """
    k = int(round(math.log2(root)))
    if 2 ** k != root:
        raise ValueError(
            f"NDB requires a power-of-two root, got {root}. Use root_method="
            "'eigh' for tensors whose ndim makes root = 2*ndim non-dyadic."
        )
    dtype = M.dtype
    n = M.shape[-1]
    A = _sym(M).float()
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    A = A + damp * _power_iteration_lmax(A) * I
    s = _power_iteration_lmax(A)
    Y = A / s
    Z = I
    for _ in range(k):
        Y, Z = _ndb_pass(Y, iters)
    # after k passes Z = (M/s)^(-1/2^k); undo the scaling
    return (Z / s ** (1.0 / root)).to(dtype)


def _contract_all(G: torch.Tensor, mats: List[Optional[torch.Tensor]],
                  index: int) -> torch.Tensor:
    """Contract dim i of G with mats[i] along `index`, for every i.

    Uses the identity that repeatedly contracting the *current* dim 0 and
    letting the new dim land at the end cycles the axes exactly once, so the
    original dim order is restored after ndim contractions. `None` entries
    are skipped via movedim, which preserves the cycle.

    index=0 applies M^T along the dim; index=1 applies M.
    """
    X = G
    for M in mats:
        if M is None:
            X = X.movedim(0, -1)
        else:
            X = torch.tensordot(X, M, dims=([0], [index]))
    return X


# --------------------------------------------------------------------------
# optimizer
# --------------------------------------------------------------------------

class OrderedShampoo(Optimizer):
    """Kronecker-factored preconditioning with configurable momentum order.

    Args:
        params: iterable of parameters or param groups.
        lr: learning rate. With graft='rms' the update has unit RMS, so lr is
            directly a per-coordinate step scale (try 1e-3 to 3e-3 for
            transformers); with graft='none' the scale is arbitrary and lr
            must be retuned per problem.
        betas: (beta1, beta2). beta1 is the momentum decay; beta2 is the decay
            on the Kronecker factor statistics. NOTE: beta2 here is the knob
            the brief proposes sweeping -- the ADOPT robustness prediction is
            that order='pre_stale' is insensitive to it.
        eps: added to the diagonal-path / eigenbasis denominator.
        weight_decay: decoupled (AdamW-style) weight decay.
        order: 'post' | 'pre' | 'pre_stale'. See module docstring.
        mode: 'shampoo' | 'eigenbasis'.
        precondition_frequency: recompute inverse roots / eigenvectors every
            this many steps. This is the *other* headline sweep in the brief:
            the 'pre' advantage should grow with this interval.
        start_preconditioning_step: run the diagonal path until this step, so
            factor statistics can warm up.
        max_precond_dim: dims larger than this are left unpreconditioned
            (identity factor), the standard blocking-off heuristic.
        clip_lambda: callable step -> clip value for order='pre_stale'.
            Defaults to ADOPT's t^(1/4). Pass None to disable.
        graft: 'rms' | 'adam' | 'none'. Fixes Shampoo's arbitrary update scale.
            'rms' normalizes the update to unit RMS. 'adam' rescales it to
            match the norm of a co-maintained diagonal Adam update (classic
            Shampoo grafting). 'none' leaves it raw.
        nesterov: apply Yin's corrected black-box Nesterov formula. Costs one
            extra parameter-sized buffer.
        nesterov_mu: the lookahead coefficient mu (held constant, so the
            mu_t vs mu_{t+1} off-by-one is moot; if you schedule it, feed
            mu_{t+1} in and track mu_t yourself).
        rotate_momentum: in 'eigenbasis' mode, rotate the momentum buffer into
            the new eigenbasis when it is refreshed instead of leaving it
            stale. SOAP leaves it; the brief argues rotating is the point.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 2e-3,
        betas=(0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        order: str = "pre_stale",
        mode: str = "shampoo",
        precondition_frequency: int = 10,
        start_preconditioning_step: int = 25,
        max_precond_dim: int = 8192,
        clip_lambda: Optional[Callable[[int], float]] = lambda s: s ** 0.25,
        graft: str = "rms",
        nesterov: bool = False,
        nesterov_mu: float = 0.9,
        rotate_momentum: bool = True,
        root_method: str = "eigh",
        one_sided: bool = False,
        precondition: bool = True,
        precond_split: int = 1,
        ndb_iters: int = 20,
        ndb_damp: float = 1e-10,
    ):
        if order not in ("post", "pre", "pre_stale"):
            raise ValueError(f"order must be post|pre|pre_stale, got {order!r}")
        if mode not in ("shampoo", "eigenbasis"):
            raise ValueError(f"mode must be shampoo|eigenbasis, got {mode!r}")
        if graft not in ("rms", "adam", "none"):
            raise ValueError(f"graft must be rms|adam|none, got {graft!r}")
        if root_method not in ("eigh", "ndb"):
            raise ValueError(f"root_method must be eigh|ndb, got {root_method!r}")
        if root_method == "ndb" and mode == "eigenbasis":
            # NDB returns the root, never the eigenvectors the SOAP path needs.
            raise ValueError("root_method='ndb' is incompatible with mode='eigenbasis'")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must be in [0,1), got {betas}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            order=order, mode=mode,
            precondition_frequency=precondition_frequency,
            start_preconditioning_step=start_preconditioning_step,
            max_precond_dim=max_precond_dim, clip_lambda=clip_lambda,
            graft=graft, nesterov=nesterov, nesterov_mu=nesterov_mu,
            rotate_momentum=rotate_momentum,
            root_method=root_method, one_sided=one_sided,
            precondition=precondition, precond_split=precond_split,
            ndb_iters=ndb_iters, ndb_damp=ndb_damp,
        )
        super().__init__(params, defaults)

    # -- state ------------------------------------------------------------

    def _init_state(self, state, p, group):
        state["step"] = 0
        state["m"] = torch.zeros_like(p)
        if group["nesterov"]:
            state["prev_update"] = torch.zeros_like(p)
        if group["graft"] == "adam":
            state["graft_m"] = torch.zeros_like(p)
            state["graft_v"] = torch.zeros_like(p)

        n_split = int(group["precond_split"])
        if n_split > 1:
            if group["mode"] == "eigenbasis":
                raise ValueError(
                    "precond_split > 1 is only implemented for mode='shampoo'"
                )
            if p.ndim < 2 or p.shape[0] % n_split != 0:
                raise ValueError(
                    f"precond_split={n_split} does not divide dim 0 of "
                    f"{tuple(p.shape)}"
                )
        state["n_split"] = n_split

        # A fused weight is preconditioned as n_split independent blocks along
        # dim 0, so the factors are sized by the *chunk*, not the whole tensor.
        dims = list(p.shape)
        if n_split > 1:
            dims[0] //= n_split

        precondable = (
            group["precondition"]
            and p.ndim >= 2
            and any(d <= group["max_precond_dim"] for d in dims)
        )
        state["precondable"] = precondable
        if precondable:
            eligible = [
                i for i, d in enumerate(dims) if d <= group["max_precond_dim"]
            ]
            if group["one_sided"]:
                # Precondition a single axis -- the smallest eligible one, which
                # is the cheapest root and (for [big, small] weights) skips the
                # factor whose eigendecomposition dominates cost entirely. Still
                # a rotation, not just a rescale, so the ordering question the
                # optimizer exists to study is unaffected.
                eligible = [min(eligible, key=lambda i: dims[i])] if eligible else []
            keep = set(eligible)
            # Indexed [chunk][axis]; n_split == 1 is the ordinary single-block
            # case, so every consumer can use one code path.
            state["M"] = [
                [
                    torch.zeros(d, d, device=p.device, dtype=p.dtype)
                    if i in keep else None
                    for i, d in enumerate(dims)
                ]
                for _ in range(n_split)
            ]
            state["roots"] = [[None] * p.ndim for _ in range(n_split)]
            state["Q"] = [[None] * p.ndim for _ in range(n_split)]
            # dim 0 may legitimately be unpreconditioned (None), so we cannot
            # use roots[0]/Q[0] to detect "never refreshed" -- track it.
            state["have_precond"] = False
        if (not precondable) or group["mode"] == "eigenbasis":
            # diagonal second moment: raw coords on the fallback path,
            # rotated coords in eigenbasis mode
            state["v"] = torch.zeros_like(p)

    # -- Kronecker factor statistics --------------------------------------

    @staticmethod
    def _chunks(x, n_split):
        return (x,) if n_split == 1 else x.chunk(n_split, dim=0)

    @classmethod
    def _apply_precond(cls, state, x, key, index):
        """Contract x with the stored operators, per block for fused weights."""
        n = state["n_split"]
        if n == 1:
            return _contract_all(x, state[key][0], index)
        return torch.cat(
            [
                _contract_all(c, state[key][k], index)
                for k, c in enumerate(x.chunk(n, dim=0))
            ],
            dim=0,
        )

    @classmethod
    def _update_factors(cls, state, grad, beta2):
        """M_i <- beta2 * M_i + (1-beta2) * contraction of grad with itself."""
        ndim = grad.ndim
        for c, g in enumerate(cls._chunks(grad, state["n_split"])):
            for i, M in enumerate(state["M"][c]):
                if M is None:
                    continue
                others = [j for j in range(ndim) if j != i]
                Mi = torch.tensordot(g, g, dims=(others, others))
                M.mul_(beta2).add_(Mi, alpha=1.0 - beta2)

    def _refresh_preconditioner(self, state, group, bias_correction2):
        """Recompute inverse roots (shampoo) or eigenvectors (eigenbasis)."""
        ndim = len(state["M"][0])
        # Shampoo's exponent is -1/(2k) where k counts the factors actually
        # applied, not the tensor's rank. With one_sided (k=1) that is -1/2,
        # i.e. full whitening on the single preconditioned axis; using 2*ndim
        # there would apply a half-strength root. This also corrects tensors
        # with a dim above max_precond_dim (e.g. the [50257, 1024] lm_head),
        # which previously got -1/4 while only one factor was active.
        n_active = sum(1 for M in state["M"][0] if M is not None)
        root = 2 * max(n_active, 1)
        state["have_precond"] = True
        if group["mode"] == "eigenbasis":
            state["_basis_change"] = [[None] * ndim for _ in state["M"]]
        for c, Ms in enumerate(state["M"]):
            for i, M in enumerate(Ms):
                if M is None:
                    continue
                Mc = M / max(bias_correction2, 1e-30)
                if group["mode"] == "shampoo":
                    use_ndb = (
                        group["root_method"] == "ndb"
                        and 2 ** int(round(math.log2(root))) == root
                    )
                    if use_ndb:
                        state["roots"][c][i] = matrix_inverse_root_ndb(
                            Mc, root, group["ndb_damp"], group["ndb_iters"]
                        )
                    else:
                        # non-dyadic root (a 3-D tensor with 3 active factors)
                        state["roots"][c][i] = matrix_inverse_root(Mc, root)
                else:
                    old = state["Q"][c][i]
                    new = matrix_eigvecs(Mc)
                    state["Q"][c][i] = new
                    # Stored as old^T @ new. _contract_all(..., index=0) applies
                    # the transpose, i.e. new^T @ old, which is exactly the map
                    # taking coordinates in the old basis to the new one.
                    state["_basis_change"][c][i] = (
                        None if old is None else old.transpose(-2, -1) @ new
                    )

    # -- grafting ---------------------------------------------------------

    @staticmethod
    def _graft_scale(direction, state, grad, group, bc1, bc2):
        g = group["graft"]
        if g == "none":
            return direction
        # NOTE: never branch on a tensor here. `if rms <= 0` forces a
        # host-device sync once per parameter per step (139 on this model),
        # which serialises the whole optimizer against the GPU. clamp_min
        # already handles the degenerate rms == 0 case.
        rms = direction.norm() / max(direction.numel() ** 0.5, 1.0)
        if g == "rms":
            return direction / rms.clamp_min(1e-16)
        # 'adam': match the norm of a co-maintained diagonal Adam update
        b1, b2 = group["betas"]
        gm, gv = state["graft_m"], state["graft_v"]
        gm.mul_(b1).add_(grad, alpha=1 - b1)
        gv.mul_(b2).addcmul_(grad, grad, value=1 - b2)
        adam_u = (gm / bc1) / ((gv / bc2).sqrt() + group["eps"])
        target = adam_u.norm()
        return direction * (target / direction.norm().clamp_min(1e-16))

    # -- the three orderings ----------------------------------------------

    def _diagonal_direction(self, state, grad, group, t):
        """Adam / LaProp / ADOPT, depending on group['order']."""
        b1, b2 = group["betas"]
        eps, order = group["eps"], group["order"]
        m, v = state["m"], state["v"]

        if order == "post":                                   # Adam
            m.mul_(b1).add_(grad, alpha=1 - b1)
            v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
            return (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)

        if order == "pre":                                    # LaProp
            v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
            n = grad / ((v / (1 - b2 ** t)).sqrt() + eps)
            m.mul_(b1).add_(n, alpha=1 - b1)
            return m / (1 - b1 ** t)

        # 'pre_stale' -- ADOPT. v is stale (excludes grad); updated after.
        n = grad / v.sqrt().clamp_min(eps)
        if group["clip_lambda"] is not None:
            c = group["clip_lambda"](t)
            n = n.clamp(-c, c)
        m.mul_(b1).add_(n, alpha=1 - b1)
        direction = m.clone()
        v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
        return direction

    def _shampoo_direction(self, state, grad, group, t):
        b1, b2 = group["betas"]
        order, freq = group["order"], group["precondition_frequency"]
        m = state["m"]

        if order == "pre_stale":
            # whiten with the PREVIOUS step's statistics, then update them
            direction = self._apply_precond(state, grad, "roots", 0)
            if group["clip_lambda"] is not None:
                c = group["clip_lambda"](t)
                direction = direction.clamp(-c, c)
            m.mul_(b1).add_(direction, alpha=1 - b1)
            direction = m.clone()
            self._update_factors(state, grad, b2)
            if t % freq == 0:
                self._refresh_preconditioner(state, group, 1 - b2 ** t)
            return direction

        # 'post' and 'pre' both see the current gradient in their statistics
        self._update_factors(state, grad, b2)
        if not state["have_precond"] or t % freq == 0:
            self._refresh_preconditioner(state, group, 1 - b2 ** t)

        if order == "post":                                   # Shampoo
            m.mul_(b1).add_(grad, alpha=1 - b1)
            return self._apply_precond(state, m / (1 - b1 ** t), "roots", 0)

        # 'pre' -- LaProp-Shampoo: integrate in whitened coordinates
        gw = self._apply_precond(state, grad, "roots", 0)
        m.mul_(b1).add_(gw, alpha=1 - b1)
        return m / (1 - b1 ** t)

    def _eigenbasis_direction(self, state, grad, group, t):
        """SOAP-style: diagonal LaProp/ADOPT/Adam inside the eigenbasis."""
        b1, b2 = group["betas"]
        eps, order = group["eps"], group["order"]
        freq = group["precondition_frequency"]
        m, v = state["m"], state["v"]

        stale = order == "pre_stale"

        if not stale:
            self._update_factors(state, grad, b2)
        if not state["have_precond"] or t % freq == 0:
            self._refresh_preconditioner(state, group, 1 - b2 ** t)
            bc = state.get("_basis_change")
            if bc is not None and group["rotate_momentum"]:
                # carry the momentum buffer into the new frame
                m.copy_(_contract_all(m, bc[0], index=0))

        # rotate gradient into the eigenbasis
        gr = self._apply_precond(state, grad, "Q", 0)

        if order == "post":
            m.mul_(b1).add_(gr, alpha=1 - b1)
            v.mul_(b2).addcmul_(gr, gr, value=1 - b2)
            dr = (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)
        elif order == "pre":
            v.mul_(b2).addcmul_(gr, gr, value=1 - b2)
            n = gr / ((v / (1 - b2 ** t)).sqrt() + eps)
            m.mul_(b1).add_(n, alpha=1 - b1)
            dr = m / (1 - b1 ** t)
        else:
            n = gr / v.sqrt().clamp_min(eps)
            if group["clip_lambda"] is not None:
                c = group["clip_lambda"](t)
                n = n.clamp(-c, c)
            m.mul_(b1).add_(n, alpha=1 - b1)
            dr = m.clone()
            v.mul_(b2).addcmul_(gr, gr, value=1 - b2)
            self._update_factors(state, grad, b2)

        # rotate back out
        return self._apply_precond(state, dr, "Q", 1)

    # -- step -------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, wd = group["lr"], group["weight_decay"]
            order, mode = group["order"], group["mode"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("OrderedShampoo does not support sparse grads")

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(state, p, group)

                # ADOPT seeds its second moment on step 1 and takes no step,
                # so the first denominator never contains its own numerator.
                if order == "pre_stale" and state["step"] == 0:
                    state["step"] = 1
                    if state["precondable"]:
                        self._update_factors(state, grad, 0.0)
                        self._refresh_preconditioner(state, group, 1.0)
                    # Seed the diagonal second moment unconditionally. Even a
                    # precondable tensor in 'shampoo' mode takes the diagonal
                    # path until start_preconditioning_step, and ADOPT there
                    # divides by v -- an unseeded v would mean dividing by eps
                    # on the first real step.
                    # Seeded in RAW coordinates because the warmup path is raw.
                    # In 'eigenbasis' mode this buffer is then reinterpreted in
                    # rotated coordinates at start_preconditioning_step -- an
                    # approximation at the handoff, and the same one SOAP makes.
                    if "v" not in state:
                        state["v"] = torch.zeros_like(p)
                    state["v"].addcmul_(grad, grad, value=1.0)
                    continue

                state["step"] += 1
                t = state["step"]
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t

                use_precond = (
                    state["precondable"]
                    and t >= group["start_preconditioning_step"]
                )

                if not use_precond:
                    if state["precondable"]:
                        # warm the factors up while stepping diagonally
                        self._update_factors(state, grad, b2)
                    if "v" not in state:
                        state["v"] = torch.zeros_like(p)
                    direction = self._diagonal_direction(state, grad, group, t)
                elif mode == "shampoo":
                    direction = self._shampoo_direction(state, grad, group, t)
                else:
                    direction = self._eigenbasis_direction(state, grad, group, t)

                direction = self._graft_scale(
                    direction, state, grad, group, bc1, bc2
                )

                update = direction.mul(-lr)

                if group["nesterov"]:
                    # Yin's corrected black-box formula:
                    #   phi_{t+1} = phi_t + (1+mu)U_t - mu*U_{t-1}
                    mu = group["nesterov_mu"]
                    prev = state["prev_update"]
                    delta = update.mul(1.0 + mu).sub_(prev, alpha=mu)
                    prev.copy_(update)
                    update = delta

                if wd != 0:
                    # Decoupled. NOTE: with nesterov=True these are lookahead
                    # parameters, and Yin argues decay belongs on the true
                    # parameters -- a known approximation here.
                    p.add_(p, alpha=-lr * wd)

                p.add_(update)

        return loss


# --------------------------------------------------------------------------
# self-test — RUN THIS FIRST, the torch paths above are unexecuted
# --------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 66)
    print("1. shape/finiteness smoke test across the full 2x3 family")
    print("=" * 66)
    for mode in ("shampoo", "eigenbasis"):
        for order in ("post", "pre", "pre_stale"):
            params = [
                torch.randn(16, 32, requires_grad=True),
                torch.randn(8, 4, 6, requires_grad=True),   # 3-D
                torch.randn(64, requires_grad=True),        # diagonal path
            ]
            opt = OrderedShampoo(
                params, lr=1e-2, mode=mode, order=order,
                precondition_frequency=3, start_preconditioning_step=4,
            )
            for _ in range(25):
                for q in params:
                    q.grad = torch.randn_like(q)
                opt.step()
            ok = all(torch.isfinite(q).all() for q in params)
            print(f"  {mode:11s} {order:10s} finite={ok}")

    print()
    print("=" * 66)
    print("2. Reddi et al. counterexample (diagonal path)")
    print("   f_t(x) = Cx if t%3==1 else -x,  x in [-1,1],  C=5")
    print("   sum of grads per cycle > 0, so the true optimum is x = -1")
    print("=" * 66)
    C, T = 5.0, 60_000
    for order in ("post", "pre", "pre_stale"):
        x = torch.zeros(1, requires_grad=True)
        opt = OrderedShampoo(
            [x], lr=1.0 / T ** 0.5, betas=(0.0, 1 / (1 + C ** 2)),
            order=order, graft="none", eps=1e-16,
        )
        for t in range(1, T + 1):
            x.grad = torch.tensor([C if t % 3 == 1 else -1.0])
            opt.step()
            with torch.no_grad():
                x.clamp_(-1, 1)
        name = {"post": "Adam", "pre": "LaProp", "pre_stale": "ADOPT"}[order]
        verdict = "converged" if x.item() < -0.5 else "DIVERGED to worst point"
        print(f"  {name:8s} ({order:10s}) x = {x.item():+.4f}   {verdict}")
    print()
    print("  Expected: Adam and LaProp both diverge to ~+1, ADOPT -> -1.")
    print("  Reordering alone does not fix non-convergence; staleness does.")

    print()
    print("=" * 66)
    print("3. do 'pre' and 'post' actually differ under matrix preconditioning?")
    print("   (if this angle is ~0 the whole research question is moot)")
    print("=" * 66)
    import math
    for freq in (1, 10, 50):
        outs = {}
        for order in ("post", "pre"):
            torch.manual_seed(1)
            w = torch.randn(24, 24, requires_grad=True)
            opt = OrderedShampoo(
                [w], lr=1e-2, order=order, mode="shampoo",
                precondition_frequency=freq, start_preconditioning_step=2,
                graft="rms",
            )
            torch.manual_seed(2)
            for _ in range(60):
                w.grad = torch.randn_like(w) @ torch.diag(
                    torch.linspace(0.1, 4.0, 24)
                )
                opt.step()
            outs[order] = w.detach().flatten()
        cos = torch.dot(outs["post"], outs["pre"]) / (
            outs["post"].norm() * outs["pre"].norm()
        )
        ang = math.degrees(math.acos(cos.clamp(-1, 1).item()))
        print(f"  precondition_frequency={freq:3d}  "
              f"cos={cos.item():+.4f}  angle={ang:5.1f} deg")
    print()
    print("  The brief predicts this angle GROWS with the refresh interval.")
    print("  That trend is the headline ablation.")
