"""GDN-3 configuration (model + bank/archive options)."""
from __future__ import annotations
import dataclasses as dc


@dc.dataclass(frozen=True)
class GDN3Config:
    # ---- model (GDN-2 backbone) ----
    vocab: int = 256
    d_model: int = 256
    n_heads: int = 4
    d_head: int = 64
    n_layers: int = 4
    mlp_mult: int = 2
    d_conv: int = 4
    pre_norm: bool = True            # RMSNorm before the mixer (same for GDN-2 baseline and GDN-3)
    g_max: float = 5.0               # clamp |g|; BTL kernel needs g_max * bs2 / 2 < 88  (bs2=32 -> g_max < 5.5)
    decay_mode: str = "channel"      # 'head' = scalar decay per head (GDN-2 report) | 'channel' = log-spaced horizons
    decay_init: float = 0.05         # 'head' mode init of softplus-rate
    hl_min: float = 16.0             # 'channel' mode: shortest half-life (tokens)
    hl_max: float = 16384.0          # 'channel' mode: longest half-life (tokens)
    # ---- banks (GDN-3 archive) ----
    seg: int = 512                   # C: tokens per bank. Rule of thumb: facts per bank <~ d_head/2  (see docs/DESIGN.md)
    bank_every: int = 2              # layers l with l % bank_every == bank_every-1 get banks; 0 = none (== GDN-2)
    bank_layers: tuple | None = None # explicit layer indices (overrides bank_every)
    gate_init: float = 0.1           # initial archive gate (tanh); 0.0 -> exactly GDN-2 at init (router then learns slowly)
    tau_init: float = 10.0           # read-norm routing sharpness
    theta_init: float = 1.0          # null-slot logit ("no bank is confident")
    bank_decay: bool = True          # decay old banks by the cumulative g since their end (faithful to the true state)
    read_precision: str = "high"     # 'default' | 'high' | 'highest'
    # ---- optional exact landmark cache (HOLA-style, experimental) ----
    n_landmarks: int = 0             # P per segment; 0 = off
    landmark_sharp: float = 8.0
    landmark_null: float = 3.0
    remat: bool = False              # jax.checkpoint per layer

    @property
    def P(self) -> int:
        return self.n_heads * self.d_head

    def has_banks(self, l: int) -> bool:
        if self.bank_layers is not None:
            return l in self.bank_layers
        return self.bank_every > 0 and (l % self.bank_every == self.bank_every - 1)

    def with_(self, **kw) -> "GDN3Config":
        return dc.replace(self, **kw)


def gdn2_like(cfg: GDN3Config) -> GDN3Config:
    """Same backbone, banks off: the GDN-2 baseline for fair comparison."""
    return cfg.with_(bank_every=0, bank_layers=(), n_landmarks=0)
