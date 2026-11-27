"""The tau discriminator that ranks variables by how exogenous they look.

causalbase's (:mod:`causalbase.lingam`), applied by the server to the pooled
cumulants.
"""

from __future__ import annotations

from causalbase.lingam.tau import tau_matrix_3rd, tau_matrix_4th

__all__ = ["tau_matrix_3rd", "tau_matrix_4th"]
