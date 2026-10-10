"""Single source of truth for GDN-3 configuration."""
import dataclasses as dc
import math

MODES = ("gdn2", "hybrid", "snap")
CELLS = ("gdn2", "qed", "eda", "dp", "osdn")
POLICIES = ("fifo", "merge", "leveled", "fenwick")   # fenwick = legacy alias of leveled (it is NOT a Fenwick tree)
DECAYS = ("head", "channel", "kda")


def _chk(v, allowed, name):
    if v not in allowed:
        raise ValueError(f"{name}={v!r} must be in {allowed}")


@dc.dataclass(frozen=True)
class G3Config:
    # ---- backbone ----
    vocab: int = 256
    d_model: int = 256
    n_heads: int = 4
    d_head: int = 64
    n_layers: int = 4
    mlp_mult: int = 2
    d_conv: int = 4
    remat: bool = False
    pad_id: int = 0
    # ---- recurrence cell (layer A) ----
    cell: str = "gdn2"            # gdn2 | qed (A3) | eda (A11) | dp (A4) | osdn (A5; scan engine only)
    dp_extra: int = 1             # extra delta steps per token for cell=dp  (n_h = 1 + dp_extra)
    osdn_eta: float = 0.01        # init of the learned per-head hypergradient step
    gamma_init: float = -2.0      # bias of sigmoid(gamma) = erase strength of qed / eda
    # ---- decay (A2) ----
    decay_mode: str = "channel"   # head | channel (log-spaced per-channel rate x per-head gate) | kda (full per-channel gate)
    decay_init: float = 0.05      # 'head' mode init of the decay rate
    hl_min: float = 16.0
    hl_max: float = 16384.0
    g_max: float = 5.0            # |g| clamp; BTL kernel needs g_max*bs2/2 < 88
    # ---- memory ----
    mode: str = "hybrid"          # gdn2 | hybrid (exact GDN-2 + independent banks) | snap (banks = snapshots of the cumulative state)
    seg: int = 128                # C: tokens per bank (fixed segmentation)
    seg_mode: str = "fixed"       # fixed | dla (B1: adaptive boundaries; scan engine)
    dla_mode: str = "residual"    # residual (||k_t - k_{t-1}||) | beta (mean beta)
    dla_min: int = 32
    dla_max: int = 0              # 0 -> 4*seg
    dla_tau: float = 0.25
    dla_theta: float = 1.0
    arch_heads: int = 0           # 0 -> all heads
    top_k: int = 2
    sketch_dim: int = -1          # JL sketch width of the router index (-1 -> D//2, 0 -> exact)
    precision: str = "high"       # matmul precision of router / read (parallel engine)
    gate_init: float = 0.1        # raw init of tanh gate on the archive read (0.1, NOT 0: router gets no gradient at 0)
    tau_init: float = 4.0
    lam_init: float = 0.1
    null_init: float = 0.5
    h_init_scale: float = 0.0     # B5: learned bank initial state, init scale
    bank_decay: bool = True       # B6: read bank j through exp(G_t - Gend_j)
    adaptive_lam: bool = False    # B12: content-dependent recency lambda per age-level
    n_levels: int = 6             # age-levels of the adaptive lambda; 0 -> auto = floor(log2(slots))+1 (see n_lv)
    lam_hidden: int = 32
    ccq: bool = False             # C1: curvature-conditioned query on the archive read
    sse_k: int = 0                # B4: read only the top-k rows (|q_i|*||S_i||) of a bank
    hola: int = 0                 # B8: ring size of the exact KV cache for surprising tokens (0 = off; scan engine)
    hola_thr: float = 0.5
    hola_gamma: float = 8.0
    # ---- archive budget (scan / streaming engine) ----
    max_banks: int = 0            # 0 -> enough slots for the whole sequence
    bank_policy: str = "fifo"     # fifo | merge (decay-correct superposition) | leveled (LSM-style log-linear compaction; alias: fenwick)
    per_level: int = 2
    max_levels: int = 12
    quant: str = "none"           # none | int8 : D1 LeapQuant storage of banks in the streaming cache
    quant_k: int = 8              # outliers kept exactly per bank

    def __post_init__(self):
        _chk(self.mode, MODES, "mode"); _chk(self.cell, CELLS, "cell"); _chk(self.bank_policy, POLICIES, "bank_policy")
        _chk(self.decay_mode, DECAYS, "decay_mode"); _chk(self.quant, ("none", "int8"), "quant")
        if self.n_levels < 0:
            raise ValueError("n_levels >= 0")
        _chk(self.seg_mode, ("fixed", "dla"), "seg_mode"); _chk(self.dla_mode, ("residual", "beta"), "dla_mode")
        if not 0 <= self.arch_heads <= self.n_heads:
            raise ValueError("0 <= arch_heads <= n_heads")
        if self.seg < 1 or self.top_k < 1 or self.dla_min < 1 or self.per_level < 2:
            raise ValueError("seg>=1, top_k>=1, dla_min>=1, per_level>=2")
        if self.max_banks == 1:
            raise ValueError("max_banks must be 0 or >= 2")
        if self.quant != "none" and self.sketch_m == 0:
            raise ValueError("quant needs a sketch router (sketch_dim != 0)")

    # ---- derived ----
    @property
    def proj(self) -> int:
        return self.n_heads * self.d_head

    @property
    def ha(self) -> int:
        return 0 if self.mode == "gdn2" else (self.arch_heads or self.n_heads)

    @property
    def m(self) -> int:
        """kernel sub-steps per token: qed/eda 2, dp 1+dp_extra, otherwise 1."""
        return {"qed": 2, "eda": 2, "dp": 1 + self.dp_extra}.get(self.cell, 1)

    @property
    def sketch_m(self) -> int:
        m = self.d_head // 2 if self.sketch_dim < 0 else self.sketch_dim
        return 0 if (m <= 0 or m >= self.d_head) else m

    @property
    def n_lv(self) -> int:
        """Effective number of lambda age-levels (age bucket = floor(log2(1+age)) <= floor(log2(#banks)))."""
        if self.n_levels > 0:
            return self.n_levels
        return max(2, int(math.log2(max(self.max_banks or 4096, 2))) + 1)

    @property
    def min_len(self) -> int:
        return self.seg if self.seg_mode == "fixed" else self.dla_min

    @property
    def max_len(self) -> int:
        return self.seg if self.seg_mode == "fixed" else (self.dla_max or 4 * self.seg)

    def slots(self, L: int) -> int:
        """Archive slots needed to keep every bank of a length-L sequence."""
        return self.max_banks or max(-(-L // self.min_len) + 1, 2)

    def parallel_ok(self) -> bool:
        return self.cell != "osdn" and self.seg_mode == "fixed" and self.hola == 0

    def align(self, bt: int) -> int:
        """Sequence-length multiple required by the parallel engine."""
        t = max(bt, 1) // math.gcd(max(bt, 1), self.m)
        return t if self.mode == "gdn2" else self.seg * t // math.gcd(self.seg, t)

    def check_backend(self, bt: int) -> None:
        if bt > 1 and self.mode != "gdn2" and (self.seg * self.m) % bt:
            raise ValueError(f"seg*m = {self.seg * self.m} must be a multiple of the kernel chunk bt={bt}")

    def with_(self, **kw) -> "G3Config":
        return dc.replace(self, **kw)


def parse_overrides(spec: str) -> dict:
    """'mode=hybrid,seg=128,ccq=true' -> dict"""
    out = {}
    for item in filter(None, (s.strip() for s in (spec or "").split(","))):
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue
        for cast in (int, float, str):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                pass
    return out
