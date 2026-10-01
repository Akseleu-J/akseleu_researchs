"""Single source of truth for GDN-3 model/memory configuration."""
from __future__ import annotations
import dataclasses as dc
import math

MODES = ("gdn2", "local", "hybrid", "snap")
POLICIES = ("fifo", "merge")
PRECISIONS = ("default", "high", "highest")


@dc.dataclass(frozen=True)
class G3Config:
    # ---- model ----
    vocab: int = 256
    d_model: int = 256
    n_heads: int = 4
    d_head: int = 64
    n_layers: int = 4
    mlp_mult: int = 2
    d_conv: int = 4
    remat: bool = True              # jax.checkpoint per layer (training memory)
    pad_id: int = 0

    # ---- memory ----
    # gdn2  : baseline. One cumulative state, no archive (exact GDN-2).
    # local : archive heads use INDEPENDENT segment states (+ warm-up W); other heads are cumulative GDN-2.
    # hybrid: ALL heads cumulative (exact GDN-2) + archive heads additionally build independent banks.
    #         => strict superset of GDN-2 (gate=0 gives GDN-2 exactly).
    # snap  : all heads cumulative; archive = snapshots of the cumulative state at segment boundaries.
    mode: str = "hybrid"
    seg: int = 512                  # C: tokens per bank (multiple of the kernel chunk, 128 on Pallas)
    warm: int = 0                   # W: warm-up tokens before the segment start (mode=local only)
    arch_heads: int = 0             # number of heads with an archive (0 -> all heads)
    top_k: int = 2                  # banks read per token
    sketch_dim: int = -1            # JL sketch width of the router index (-1 -> d_head//2, 0 -> exact)
    precision: str = "high"         # matmul precision of router/read (bf16x3)
    gate_init: float = 0.1          # raw init of tanh gate on the archive read (0.1, NOT 0: see docs)
    tau_init: float = 4.0           # router temperature
    lam_init: float = 0.1           # recency prior strength
    null_init: float = 0.5          # "read nothing" logit (in units of tau)

    # ---- decay / numerics ----
    hl_min: float = 16.0            # shortest decay half-life (tokens) per channel
    hl_max: float = 16384.0         # longest decay half-life
    g_max: float = 5.0              # clamp of |g|; must match the kernel cfg (160/bs2)

    # ---- streaming archive ----
    bank_policy: str = "fifo"       # what to do when max_banks is exceeded: fifo | merge (sink bank)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be in {MODES}")
        if self.bank_policy not in POLICIES:
            raise ValueError(f"bank_policy must be in {POLICIES}")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be in {PRECISIONS}")
        if not 0 <= self.arch_heads <= self.n_heads:
            raise ValueError("0 <= arch_heads <= n_heads")
        if self.seg <= 0:
            raise ValueError("seg > 0")
        if not 0 <= self.warm < self.seg:
            raise ValueError("0 <= warm < seg")
        if self.mode != "local" and self.warm:
            raise ValueError("warm > 0 is only meaningful for mode='local'")
        if self.top_k < 1:
            raise ValueError("top_k >= 1")

    # ---- derived ----
    @property
    def proj(self) -> int:
        return self.n_heads * self.d_head

    @property
    def ha(self) -> int:
        """Number of archive heads (0 for the gdn2 baseline)."""
        if self.mode == "gdn2":
            return 0
        return self.arch_heads or self.n_heads

    @property
    def bank_warm(self) -> int:
        return self.warm if self.mode == "local" else 0

    @property
    def sketch_m(self) -> int:
        m = self.d_head // 2 if self.sketch_dim < 0 else self.sketch_dim
        return 0 if (m <= 0 or m >= self.d_head) else m

    def align(self, bt: int) -> int:
        """Sequence-length multiple required by the forward pass."""
        bt = max(bt, 1)
        if self.mode == "gdn2":
            return bt
        return self.seg * bt // math.gcd(self.seg, bt)

    def check_backend(self, bt: int) -> None:
        if self.mode == "gdn2" or bt <= 1:
            return
        if self.seg % bt or (self.seg + self.bank_warm) % bt:
            raise ValueError(f"seg={self.seg}, warm={self.bank_warm} must be multiples of the kernel chunk bt={bt}")

    def with_(self, **kw) -> "G3Config":
        return dc.replace(self, **kw)

    def n_params_estimate(self) -> int:
        d, P, H, D = self.d_model, self.proj, self.n_heads, self.d_head
        per = 5 * d * P + d * H + 2 * P + P * d + 3 * self.mlp_mult * d * d + H * D
        if self.ha:
            per += self.ha * D * D + 4 * self.ha
        return self.vocab * d + self.n_layers * per


def parse_overrides(spec: str) -> dict:
    """'mode=hybrid,seg=128,top_k=2' -> {'mode': 'hybrid', 'seg': 128, 'top_k': 2}"""
    out = {}
    if not spec:
        return out
    for item in spec.split(","):
        if not item.strip():
            continue
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out
