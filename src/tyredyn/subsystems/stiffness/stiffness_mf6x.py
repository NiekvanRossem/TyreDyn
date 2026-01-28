from tyredyn.types.aliases import SignalLike
from tyredyn.infrastructure.subsystem_base import SubSystemBase
import numpy as np
from typing import Literal

class StiffnessMF6x(SubSystemBase):
    """
    Stiffness module for the MF-Tyre 6.1 and MF-Tyre 6.2 models.
    """

    def _connect(self, model):
        self.normalize = model.normalize

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
