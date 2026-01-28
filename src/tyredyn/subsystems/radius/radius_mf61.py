from tyredyn.types.aliases import NumberLike, SignalLike
from .radius_mf6x import RadiusMF6x
from typing import Literal
import numpy as np
import warnings

class RadiusMF61(RadiusMF6x):
    """
    Radius and deflection module for the MF-Tyre 6.1 model.
    """

    def _find_radius(
            self,
            *,
            FX: SignalLike,
            FY: SignalLike,
            FZ: SignalLike,
            N:  SignalLike,
            P:  SignalLike = None
    ) -> list[SignalLike]:
        """
        Returns the various radii and deflection of the tyre. Order is ``R_omega``, ``RE``, ``RL``, ``rho_z``.

        Parameters
        ----------
        FZ : SignalLike
            Vertical load.
        N : SignalLike, optional
            Angular speed of the wheel (will be calculated from ``VX`` and ``SL`` if not specified).
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).

        Returns
        -------
        R_omega, RE, RL, rho_z : list[SignalLike]
            Free rolling radius, effective radius, loaded radius, and vertical deflection.
        """

        # unpack tyre properties
        CZ0 = self.VERTICAL_STIFFNESS
        FZ0 = self.FNOMIN
        R0  = self.UNLOADED_RADIUS
        V0  = self.LONGVL

        # _normalize tyre pressure
        dpi = self.normalize._find_dpi(P)

        # free rolling radius
        R_omega = self._find_free_radius(N=N, R0=R0, V0=V0)

        # effective radius
        RE = self._find_effective_radius(FZ=FZ, P=P, R_omega=R_omega, FZ0=FZ0)

        # find QFZ1
        Q_FZ1 = self._find_qfz1(CZ0=CZ0, R0=R0, FZ0=FZ0)

        # loaded radius
        RL = self._converge_loaded_radius(FX=FX, FY=FY, FZ=FZ, N=N, dpi=dpi, IA=None, R_omega=R_omega, Q_FZ1=Q_FZ1,
                                          FZ0=FZ0, R0=R0, V0=V0)

        # vertical deflection
        rho_z = np.maximum(R_omega - RL, 1e-12)

        return [R_omega, RE, RL, rho_z]

    def _find_fz(
            self,
            *,
            FX:      SignalLike,
            FY:      SignalLike,
            RL:      SignalLike,
            R_omega: SignalLike,
            N:       SignalLike,
            dpi:     SignalLike,
            FZ0:     NumberLike,
            R0:      NumberLike,
            V0:      NumberLike,
            Q_FZ1:   NumberLike,
            **kwargs
    ) -> SignalLike:
        """Returns the vertical load as a function of the free rolling radius and the loaded radius."""

        # unpack tyre properties
        CZ_btm    = self.BOTTOM_STIFF
        R_rim     = self.RIM_RADIUS
        delta_btm = self.BOTTOM_OFFST

        # vertical deflection
        rho_z = R_omega - RL

        # vertical load (A3.3 from the 2012 book by Pacejka & Besselink)
        speed_effect    = self.Q_V2 * np.abs(N) * R0 / V0
        fx_effect       = (self.Q_FCX * FX / FZ0) ** 2
        fy_effect       = (self.Q_FCY * FY / FZ0) ** 2
        pressure_effect = 1.0 + self.PFZ1 * dpi
        residual_effect = Q_FZ1 * rho_z / R0 + self.Q_FZ2 * (rho_z / R0) ** 2
        FZ = (1.0 + speed_effect - fx_effect - fy_effect) * residual_effect * pressure_effect * FZ0

        # NOTE: The equation above is taken from the appendix of the 2012 book by Pacejka & Besselink. Equation (4.E68)
        # adds a camber term to it (shown below), but Q_FZ3 is not a standard term in TIR files, nor is an equation
        # provided to calculate it.
        # residual_effect = (Q_FZ1 + Q_FZ3 * IA ** 2) * rho_z / R0 + Q_FZ2 * (rho_z / R0) ** 2

        # bottoming out force (MF-Tyre 6.2 equation manual)
        FZ_btm = CZ_btm * (R_rim + delta_btm - RL)

        # total vertical load
        FZ_final = np.maximum(FZ, FZ_btm)
        return FZ_final
