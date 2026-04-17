"""Report generation: plot + save for FitResult objects.

All plotting is opt-in here (not inside estimators). Every plot function
returns ``(fig, ax)`` for inline Jupyter display; writes to disk only if
``path=`` is given.
"""

from cmb_diagnostics.reports import pol_angle, tf

__all__ = ["pol_angle", "tf"]
