"""Compatibility wrapper for the legacy :mod:`waven.zebraGUI` module.

New code should import GUI entry points from :mod:`waven.app.gui`.
"""

from .app.gui import *  # noqa: F401,F403
