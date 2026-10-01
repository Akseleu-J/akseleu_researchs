"""GDN-3: Gated DeltaNet-2 with a dynamic archive of independent state matrices ("banks")."""
from .config import GDN3Config, gdn2_like
from .backend import make_backend, make_ref_backend, ensure_atomic_path
from .model import init_model, forward, lm_loss, count_params
from .stream import StreamingLM
__version__ = "0.1.0"
