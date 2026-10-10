# Atomic donor and patches

Source: the user-supplied atomic_ops/pkg/atomic_gdn2 snapshot. The original files are not modified. See the root audit manifest for SHA-256 provenance. No upstream license file was supplied with this package; no third-party license is invented here.

- R20: mega VJP retains decay contributions at g=0.
- R21: lean diagonal VJP differentiates the forward clipped pair exponent. The chosen exact-boundary derivative is zero; compare finite differences away from the kink.
- R23: correctness preset uses WY epsilon zero.
- Positive geometry validation and empty-sequence behavior added.
- Legacy sanitization remains a declared surrogate outside its healthy domain. The new v2 asymmetric Pallas primitive has an analytical backward without those clamps.

These patches are authored, not hardware-validated. Step 4 contains the required TPU checks.
