from src.utils.formatting import SignalLike
import numpy as np
from typing import Literal

class StiffnessMF61:

    def __init__(self, model):
        """Import the properties of the overarching ``MF61`` class."""
        self._model = model

        # helper functions
        self.normalize  = model.normalize

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def find_lateral_stiffness(
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

        # set default value for optional arguments
        P = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, P = self._format_check([FZ, P])

        # unpack tyre properties
        Cy0 = self.LATERAL_STIFFNESS

        # normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)
        dpi = self.normalize._find_dpi(P)

        # lateral stiffness (A3.10)
        Cy = Cy0 * (1.0 + self.PCFY1 * dfz + self.PCFY2 * dfz ** 2) * (1.0 + self.PCFY3 * dpi)
        return Cy

    def find_longitudinal_stiffness(
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

        # set default value for optional arguments
        P = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, P = self._format_check([FZ, P])

        # unpack tyre properties
        Cx0 = self.LONGITUDINAL_STIFFNESS

        # normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)
        dpi = self.normalize._find_dpi(P)

        # lateral stiffness (A3.10)
        Cx = Cx0 * (1.0 + self.PCFX1 * dfz + self.PCFX2 * dfz ** 2) * (1.0 + self.PCFX3 * dpi)
        return Cx

    def find_vertical_stiffness(
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

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            P = self._format_check(P)

        # unpack tyre properties
        CZ0 = self.VERTICAL_STIFFNESS

        # normalize pressure
        dpi = self.normalize._find_dpi(P)

        # current vertical rate (A3.5)
        CZ = CZ0 * (1.0 + self.PFZ1 * dpi)

        return CZ
