from .constitutive_sma import NiTiParams, newton_solve, niti_update
from .geometry import canopy_geometry
from .niti import NiTiParams as LegacyNiTiParams
from .niti import helmholtz_free_energy as legacy_helmholtz_free_energy
from .niti import niti_update as legacy_niti_update
from .solver import newton_solve as legacy_newton_solve
from .thermodynamics import discrete_dissipation, helmholtz_free_energy

__all__ = [
    "NiTiParams",
    "canopy_geometry",
    "discrete_dissipation",
    "helmholtz_free_energy",
    "newton_solve",
    "niti_update",
]
