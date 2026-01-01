from src.utils.formatting import SignalLike
import numpy as np
from typing import Literal

class StiffnessMF6x:
    """
    Stiffness module for the MF 6.1 and MF 6.2 tyre models.
    """

    def __init__(self, model):
        """Import the properties of the overarching ``MF61`` or ``MF62`` class."""
        self._model = model

        # helper functions
        self.normalize  = model.normalize

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def _find_lateral_stiffness(
            self,
            *,
            FZ: SignalLike,
            P:  SignalLike = None
    ) -> SignalLike:
        """
        Returns the lateral stiffness of the tyre, adjusted for load and pressure.

        Parameters
        ----------
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).

        Returns
        -------
        Cy : SignalLike
            Lateral tyre stiffness.
        """

        # unpack tyre properties
        Cy0 = self.LATERAL_STIFFNESS

        # _normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)
        dpi = self.normalize._find_dpi(P)

        # lateral stiffness (A3.10)
        Cy = Cy0 * (1.0 + self.PCFY1 * dfz + self.PCFY2 * dfz ** 2) * (1.0 + self.PCFY3 * dpi)
        return Cy

    def _find_longitudinal_stiffness(
            self,
            *,
            FZ: SignalLike,
            P:  SignalLike = None
    ) -> SignalLike:
        """
        Returns the longitudinal stiffness of the tyre, adjusted for load and pressure.

        Parameters
        ----------
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).

        Returns
        -------
        Cx : SignalLike
            Longitudinal tyre stiffness.
        """

        # unpack tyre properties
        Cx0 = self.LONGITUDINAL_STIFFNESS

        # _normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)
        dpi = self.normalize._find_dpi(P)

        # lateral stiffness (A3.10)
        Cx = Cx0 * (1.0 + self.PCFX1 * dfz + self.PCFX2 * dfz ** 2) * (1.0 + self.PCFX3 * dpi)
        return Cx

    def _find_vertical_stiffness(
            self,
            P: SignalLike
    ) -> SignalLike:
        """
        Returns the vertical stiffness of the tyre, adjusted for pressure.

        Parameters
        ----------
        P : SignalLike
            Tyre pressure.

        Returns
        -------
        Cz : SignalLike
            Vertical tyre stiffness.
        """

        # unpack tyre properties
        CZ0 = self.VERTICAL_STIFFNESS

        # _normalize pressure
        dpi = self.normalize._find_dpi(P)

        # current vertical rate (A3.5)
        CZ = CZ0 * (1.0 + self.PFZ1 * dpi)

        return CZ
