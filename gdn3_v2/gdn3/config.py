"""Validated, serializable contracts; configuration never silently disables a feature."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace
import hashlib
import json
import math

@dataclass(frozen=True)
class PrecisionPolicy:
    dot: str = "highest"
    storage: str = "float32"
    accumulation: str = "float32"
    wy_eps: float = 0.0
    diff_clip: float | None = None
    strict_health: bool = True
    def __post_init__(self):
        if self.dot not in {"highest", "bf16x3", "default"}:
            raise ValueError("dot must be highest, bf16x3 or default")
        if self.storage not in {"float32", "bfloat16"} or self.accumulation != "float32":
            raise ValueError("supported storage: float32/bfloat16; accumulation: float32")
        if not 0 <= self.wy_eps < 1:
            raise ValueError("wy_eps must be in [0,1)")
        if self.diff_clip is not None and self.diff_clip <= 0:
            raise ValueError("diff_clip must be positive")

@dataclass(frozen=True)
class GDN3Config:
    vocab: int = 256
    d_model: int = 128
    n_heads: int = 4
    d_key: int = 32
    d_value: int = 32
    n_layers: int = 2
    mlp_mult: int = 2
    d_conv: int = 4
    mode: str = "hybrid"
    backbone: str = "kit"
    archive_heads: tuple[int, ...] = ()
    memory: str = "hierarchy"
    recent_slots: int = 8
    recent_horizon: int = 512
    initial_levels: int = 4
    max_levels: int = 0
    initial_counter_words: int = 2
    top_k: int = 2
    router_dim: int = 16
    gate_init: float = 0.1
    router_temperature: float = 4.0
    router_age: float = 0.0
    router_null: float = 0.5
    learned_init: bool = True
    bank_decay: str = "frozen"
    segment_min: int = 16
    segment_max: int = 128
    dla: bool = True
    dla_threshold: float = 0.3
    boundary_grad: str = "straight_through"
    boundary_temperature: float = 0.1
    warm: int = 0
    decay_mode: str = "kda"
    decay_rank: int = 16
    hl_min: float = 16.0
    hl_max: float = 16384.0
    g_max: float = 5.0
    cell: str = "gdn2"
    qed: bool = False
    qed_init: float = 0.1
    eda: bool = False
    eda_init: float = 0.1
    delta_steps: int = 1
    osdn: str = "off"
    osdn_lr: float = 0.01
    osdn_min: float = 0.25
    osdn_max: float = 2.0
    osdn_margin: float = 0.01
    sse_partitions: int = 1
    sse_top_k: int = 1
    sse_shared: bool = True
    sse_inactive: str = "freeze"
    sse_balance: float = 0.01
    lambda_mode: str = "shared"
    lambda_hidden: int = 16
    lambda_paper_levels: int = 16
    ccq: str = "archive"
    ccq_init: float = -3.0
    hola_size: int = 32
    hola_block: int = 16
    hola_gate_init: float = 0.1
    quant: str = "none"
    quant_window: int = 8
    quant_rank: int = 1
    quant_bits: int = 8
    quant_scope: str = "archive"
    oja_rank: int = 8
    oja_anchors: int = 8
    external_candidates: int = 16
    external_gate_init: float = 0.1
    pdn_x: float = 1.5
    backend: str = "jax"
    pallas_interpret: bool = False
    remat: bool = False
    norm_group_key: int = 0
    norm_group_value: int = 0
    layer_overrides: tuple[dict, ...] = ()
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)

    def __post_init__(self):
        object.__setattr__(self, "archive_heads", tuple(self.archive_heads))
        if isinstance(self.precision, dict):
            object.__setattr__(self, "precision", PrecisionPolicy(**self.precision))
        positive = ("vocab", "d_model", "n_heads", "d_key", "d_value", "n_layers", "mlp_mult", "d_conv", "recent_slots", "recent_horizon", "initial_levels", "initial_counter_words", "top_k", "router_dim", "segment_min", "segment_max", "decay_rank", "delta_steps", "sse_partitions", "sse_top_k", "lambda_hidden", "lambda_paper_levels", "quant_window", "external_candidates")
        for name in positive:
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        enums = {"mode": {"gdn2", "hybrid", "local", "snap"}, "backbone": {"kit", "atomic"}, "memory": {"hierarchy", "loglinear", "fifo", "none"}, "bank_decay": {"frozen", "aligned"}, "boundary_grad": {"fixed", "soft", "straight_through"}, "decay_mode": {"head", "channel", "kda", "kda_lowrank", "atomic"}, "cell": {"gdn2", "fg2", "qdelta", "pdn"}, "osdn": {"off", "paper", "generalized"}, "lambda_mode": {"off", "shared", "paper_eq4", "paper_gelu"}, "ccq": {"off", "archive", "all"}, "quant": {"none", "leap", "archive_int8"}, "quant_scope": {"archive", "all"}, "sse_inactive": {"freeze", "passive_lazy"}, "backend": {"jax", "pallas", "auto"}}
        for name, choices in enums.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {sorted(choices)}")
        if self.recent_slots < 2:
            raise ValueError("recent_slots >= 2 is required for adjacent merge")
        if not 0 < self.segment_min <= self.segment_max:
            raise ValueError("0 < segment_min <= segment_max")
        if not 0 < self.hl_min <= self.hl_max or self.g_max <= 0:
            raise ValueError("invalid half-lives/g_max")
        if any(h < 0 or h >= self.n_heads for h in self.archive_heads) or len(set(self.archive_heads)) != len(self.archive_heads):
            raise ValueError("archive_heads must be unique valid head indices")
        if self.max_levels < 0 or self.max_levels and self.max_levels < self.initial_levels:
            raise ValueError("max_levels must be zero or >= initial_levels")
        if self.sse_top_k > self.sse_partitions:
            raise ValueError("sse_top_k exceeds partitions")
        if self.sse_partitions > 1 and self.mode in {"gdn2", "snap"}:
            raise ValueError("SSE expands independent archive banks only")
        if self.warm < 0 or self.warm >= self.segment_min or self.warm and self.mode != "local":
            raise ValueError("warm requires local mode and 0 <= warm < segment_min")
        if self.warm and (self.osdn != "off" or self.sse_partitions > 1 or self.quant != "none"):
            raise ValueError("warm replay supports unquantized cells without OSDN/SSE")
        if self.memory == "loglinear" and (self.mode != "hybrid" or self.warm or self.quant != "none" or self.sse_partitions > 1):
            raise ValueError("loglinear supports hybrid, unquantized, no warm/SSE; choose hierarchy for these adapters")
        if self.mode == "snap" and (self.memory != "fifo" or self.top_k != 1):
            raise ValueError("snap requires FIFO and top_k=1 to avoid overlapping snapshot summation")
        if self.mode != "gdn2" and self.memory == "none":
            raise ValueError("memory=none requires gdn2 mode")
        if self.cell != "gdn2" and (self.qed or self.eda or self.osdn != "off" or self.delta_steps != 1):
            raise ValueError("deferred cells are separate cell families")
        if self.osdn == "paper" and (self.qed or self.eda or self.delta_steps != 1 or self.sse_partitions > 1):
            raise ValueError("scalar-paper OSDN is an isolated reference preset")
        if not 0 < self.osdn_min <= 1 <= self.osdn_max or self.osdn_lr < 0 or not 0 < self.osdn_margin < 1:
            raise ValueError("OSDN box must contain 1; eta>=0; 0<margin<1")
        if self.qed_init < 0 or not 0 <= self.eda_init <= 1:
            raise ValueError("invalid QED/EDA strength")
        if self.hola_size < 0 or self.hola_block < 0 or self.hola_size and not self.hola_block:
            raise ValueError("HOLA requires a positive bounded current block")
        if self.quant_bits not in {4, 8} or not 0 <= self.quant_rank <= min(self.d_key, self.d_value):
            raise ValueError("quant bits in {4,8}, rank within state dimensions")
        if self.quant != "none" and self.boundary_grad == "soft":
            raise ValueError("quantization is inference-only, soft training is not a quantized mode")
        if self.oja_rank < 1 or self.oja_anchors < 0 or not 1 <= self.pdn_x <= 2:
            raise ValueError("invalid Oja rank/anchors or PDN x")
        if self.router_temperature <= 0 or self.router_age < 0 or self.boundary_temperature <= 0 or self.dla_threshold < 0:
            raise ValueError("invalid router/boundary parameters")
        if self.lambda_mode.startswith("paper") and (not self.max_levels or self.max_levels > self.lambda_paper_levels):
            raise ValueError("paper level-axis MLP requires explicit max_levels <= lambda_paper_levels")
        if self.precision.wy_eps or self.precision.diff_clip is not None:
            raise ValueError("stream ABI is exact recurrence; WY/clipped variants belong to atomic diagnostic APIs")
        if self.backbone == "atomic" and (self.d_key != self.d_value or self.decay_mode != "atomic"):
            raise ValueError("atomic import requires square heads and atomic gate")
        for d, group in [(self.d_key, self.norm_group_key), (self.d_value, self.norm_group_value)]:
            if group < 0 or group and d % group:
                raise ValueError("normalization group must divide head dimension")
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(name + " must be finite")

    def for_layer(self, index):
        if not self.layer_overrides:return self
        if len(self.layer_overrides)!=self.n_layers:raise ValueError("one layer override per layer required")
        return replace(self,layer_overrides=(),**self.layer_overrides[index])

    @property
    def heads(self):
        return () if self.mode == "gdn2" else (self.archive_heads or tuple(range(self.n_heads)))
    @property
    def ha(self): return len(self.heads)
    @property
    def key_width(self): return self.n_heads * self.d_key
    @property
    def value_width(self): return self.n_heads * self.d_value
    @property
    def substeps(self): return self.delta_steps + int(self.eda)
    def with_(self, **kwargs): return replace(self, **kwargs)
    def to_dict(self): return asdict(self)
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    @classmethod
    def from_dict(cls, d): return cls(**d)
    @classmethod
    def baseline(cls, **kw):
        values = dict(mode="gdn2", memory="none", dla=False, decay_mode="channel", lambda_mode="off", ccq="off", hola_size=0, learned_init=False)
        values.update(kw)
        return cls(**values)
