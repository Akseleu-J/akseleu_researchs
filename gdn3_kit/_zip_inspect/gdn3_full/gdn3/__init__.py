"""GDN-3: Gated DeltaNet-2 + a dynamic pool of state matrices ("banks"). Pure JAX; Pallas kernels plug in via backend.py."""
from .config import G3Config
__version__ = "0.2.0"
__all__ = ["G3Config"]
