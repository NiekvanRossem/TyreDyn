from src.utils.formatting import SignalLike, AngleUnit
from src.helpers.corrections_mf6x import CorrectionsMF6x
from src.helpers.normalize import Normalize
from typing import Literal

class FrictionMF6x:
    """
    Friction coefficient module for the MF 6.1 and MF 6.2 tyre models.
    """

    def __init__(self, model):
        """Import the properties of the overarching ``MF61`` or ``MF62`` class."""
        self._model     = model

        # helper functions
        self.correction = model.correction
        self.normalize  = model.normalize

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def _find_mu_x(
            self,
            *,
            SA: SignalLike,
            SL: SignalLike,
            FZ: SignalLike,
            P:  SignalLike = None,
            IA: SignalLike = 0.0,
            VX: SignalLike = None
    ) -> SignalLike:
        """
        Returns the longitudinal friction coefficient.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        SL : SignalLike
            Slip ratio.
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to ``0.0`` if not specified).
        VX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).

        Returns
        -------
        mu_x : SignalLike
            Longitudinal friction coefficient.
        """

        # find other velocity components
        VS, VC = self.normalize._find_speeds(SA=SA, SL=SL, VX=VX)

        # unpack tyre properties
        V0 = self.LONGVL

        # _normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)
        dpi = self.normalize._find_dpi(P)

        # composite friction scaling factor (4.E7)
        LMUX_star = self.correction._find_lmu_star(VS=VS, V0=V0, LMU=self.LMUX)

        # friction coefficient (4.E13)
        mu_x = ((self.PDX1 + self.PDX2 * dfz) * (1.0 + self.PPX3 * dpi + self.PPX4 * dpi ** 2)
                * (1.0 - self.PDX3 * IA ** 2) * LMUX_star)

        return mu_x

    def _find_mu_y(
            self,
            *,
            SA: SignalLike,
            SL: SignalLike,
            FZ: SignalLike,
            P:  SignalLike = None,
            IA: SignalLike = 0.0,
            VX: SignalLike = None
    ) -> SignalLike:
        """
        Returns the lateral friction coefficient.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        SL : SignalLike
            Slip ratio.
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to ``0.0`` if not specified).
        VX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).

        Returns
        -------
        mu_y : SignalLike
            Lateral friction coefficient.
        """

        # find other velocity components
        VS, VC = self.normalize._find_speeds(SA=SA, SL=SL, VX=VX)

        # unpack tyre properties
        V0 = self.LONGVL

        # _normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)
        dpi = self.normalize._find_dpi(P)

        # corrected camber angle
        gamma_star = self.correction._find_gamma_star(IA)

        # composite friction scaling factor (4.E7)
        LMUY_star = self.correction._find_lmu_star(VS=VS, V0=V0, LMU=self.LMUY)

        # lateral friction coefficient (4.E23)
        mu_y = ((self.PDY1 + self.PDY2 * dfz) * (1.0 + self.PPY3 * dpi + self.PPY4 * dpi ** 2)
                * (1.0 - self.PDY3 * gamma_star ** 2) * LMUY_star)

        return mu_y
