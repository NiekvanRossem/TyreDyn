from tyredyn.types.aliases import SignalLike, AngleUnit
from typing import Literal
import numpy as np

class RelaxationMF6x:
    """
    Relaxation length module for the MF 6.1 and MF 6.2 tyre models.
    """

    def __init__(self, model):
        """Import the properties of the overarching ``MF61`` or ``MF62`` class."""
        self._model = model

        # other subsystems
        self.stiffness  = model.stiffness
        self.gradient   = model.gradient

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def _find_lateral_relaxation(
            self,
            *,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            N:    SignalLike,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0
    ) -> SignalLike:
        """
        Returns the lateral relaxation length.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        SL : SignalLike
            Slip ratio.
        FZ : SignalLike
            Vertical load.
        N : SignalLike, optional
            Angular speed of the wheel (will be calculated from ``VX`` and ``SL`` if not specified).
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to ``0.0`` if not specified).
        VX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to ``0.0`` if not specified).

        Returns
        -------
        sigma_y : SignalLike
            Lateral relaxation length.
        """

        # cornering stiffness
        KYA = self.gradient._find_cornering_stiffness(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

        # lateral stiffness
        Cy = self.stiffness._find_lateral_stiffness(FZ=FZ, P=P)

        # lateral relaxation length (A3.9)
        sigma_y = KYA / Cy
        return sigma_y

    def _find_longitudinal_relaxation(
            self,
            *,
            FZ: SignalLike,
            P:  SignalLike = None
    ) -> SignalLike:
        """
        Returns the longitudinal relaxation length.

        Parameters
        ----------
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).

        Returns
        -------
        sigma_x : SignalLike
            Longitudinal relaxation length.
        """

        # set default values for optional arguments
        #P = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        #if self._check_format:
        #    FZ, P = self.data_checks._format_check([FZ, P])

        # slip stiffness
        KXK = self.gradient._find_slip_stiffness(FZ=FZ, P=P)

        # longitudinal stiffness
        Cx = self.stiffness._find_longitudinal_stiffness(FZ=FZ, P=P)

        # longitudinal relaxation length (A3.9)
        sigma_x = KXK / Cx
        return sigma_x
