from tyredyn.types.aliases import NumberLike, SignalLike
from typing import Literal
import numpy as np
import warnings

from tyredyn.subsystems.radius.radius_mf6x import RadiusMF6x

class RadiusMF62(RadiusMF6x):
    """
    Radius and deflection module for the MF 6.2 tyre model.
    """

    def __init__(self, model):
        super().__init__(model)

    #def __init__(self, model):
        #"""Import the properties of the overarching class."""
        #self._model = model

        # helper functions
        #self.normalize  = model.normalize

        # other subsystems used
        #self.stiffness  = model.stiffness

    #def __getattr__(self, name):
        #"""Make the tyre coefficients directly available."""
        #return getattr(self._model, name)

    def _find_radius(
            self,
            *,
            FX: SignalLike,
            FY: SignalLike,
            FZ: SignalLike,
            N:  SignalLike,
            P:  SignalLike = None,
            IA: SignalLike = 0.0,
            maxiter: int   = 30,
            tolx: float    = 1e-6
    ) -> list[SignalLike]:
        """
        Returns the various radii and deflection of the tyre. Order is ``R_omega``, ``RE``, ``RL``, ``rho``.

        Parameters
        ----------
        FX : SignalLike
            Longitudinal force.
        FY : SignalLike
            Side force.
        FZ : SignalLike
            Vertical load.
        N : SignalLike, optional
            Angular speed of the wheel (will be calculated from ``VX`` and ``SL`` if not specified).
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to ``0.0`` if not specified).
        maxiter : int, optional
            Maximum number of iterations for finding ``RL``.
        tolx : float, optional
            Tolerance for the optimization.

        Returns
        -------
        R_omega, RE, RL, rho : List[SignalLike]
            Free rolling radius, effective radius, loaded radius, and vertical deflection.
        """

        # unpack tyre properties
        CZ0 = self.VERTICAL_STIFFNESS
        FZ0 = self.FNOMIN
        R0  = self.UNLOADED_RADIUS
        V0  = self.LONGVL

        # _normalize pressure
        dpi = self.normalize._find_dpi(P)

        # free rolling radius (MF6.2 equation manual)
        R_omega = self._find_free_radius(N=N, R0=R0, V0=V0)

        # effective radius (A3.6) -- FZ trig functions do not get corrected to degrees
        RE = self._find_effective_radius(FZ=FZ, P=P, R_omega=R_omega, FZ0=FZ0)

        # find QFZ1 from CZ0 (MF 6.2 equation manual)
        #Q_FZ1 = np.sqrt((CZ0 * R0 / FZ0) ** 2 - 4 * self.Q_FZ2)
        Q_FZ1 = self._find_qfz1(CZ0=CZ0, R0=R0, FZ0=FZ0)

        # set bounds for optimization
        #RL_lower  = 0.95 * R_omega
        #RL_upper  = R_omega
        #RL_newest = None

        # evaluate function for initial bounds
        #y_lower = self._find_fz(FX=FX, FY=FY, RL=RL_lower, R_omega=R_omega, N=N, dpi=dpi, IA=IA, FZ0=FZ0, R0=R0, V0=V0, Q_FZ1=Q_FZ1) - FZ
        #y_upper = self._find_fz(FX=FX, FY=FY, RL=RL_upper, R_omega=R_omega, N=N, dpi=dpi, IA=IA, FZ0=FZ0, R0=R0, V0=V0, Q_FZ1=Q_FZ1) - FZ

        # counter
        #counter = 0

        # perform optimization
        #for iteration in range(maxiter):

            # make new guess for RL
            #RL_newest = RL_upper - (RL_upper - RL_lower) / (y_upper - y_lower) * y_upper
            #y_newest = self._find_fz(FX=FX, FY=FY, RL=RL_newest, R_omega=R_omega, N=N, dpi=dpi, IA=IA, FZ0=FZ0, R0=R0,
            #                          V0=V0, Q_FZ1=Q_FZ1) - FZ

            # update values
            #y_upper = y_newest
            #RL_upper = RL_newest

            # check if all values have converged
            #error = abs(y_newest).max()
            #if error < tolx * FZ0:
            #    break
            #if counter == maxiter - 1:
            #    warnings.warn("Maximum number of iterations reached. No solution for the loaded radius found. Final error is {0}".format(error))
            #counter += 1
        #RL = RL_newest

        RL = self._converge_loaded_radius(FX=FX, FY=FY, FZ=FZ, N=N, dpi=dpi, IA=IA, R_omega=R_omega, Q_FZ1=Q_FZ1, FZ0=FZ0, R0=R0, V0=V0)

        # vertical deflection
        rho_z = self.__find_deflection(R_omega=R_omega, RL=RL, IA=IA)

        return [R_omega, RE, RL, rho_z]

    #------------------------------------------------------------------------------------------------------------------#
    # INTERNAL FUNCTIONS

    def __find_deflection(self, *, R_omega, RL, IA):
        """Finds the deflection from the free rolling radius, the loaded radius, and the inclination angle."""

        # free rolling deflection
        rho_zfr = np.maximum(R_omega - RL, 0.0)

        # reference thread width TODO: find original source
        rtw = (1.075 - 0.5 * self.ASPECT_RATIO) * self.WIDTH

        # deflection due to camber (MF 6.2 equation manual)
        numerator = ((self.Q_CAM1 * RL + self.Q_CAM2 * RL ** 2) * IA) ** 2 * (rtw / 8.0 * np.abs(self.tan(IA)))
        denominator = ((self.Q_CAM1 * R_omega + self.Q_CAM2 * R_omega ** 2) * IA + 1e-12) ** 2 # TODO replace nan values with zero
        rho_zc = numerator / denominator - self.Q_CAM3 * rho_zfr * np.abs(IA)

        # total vertical deflection (MF 6.2 equation manual)
        rho = np.maximum(rho_zfr + rho_zc, 1e-12)
        return rho

    def _find_fz(
            self,
            *,
            FX:      SignalLike,
            FY:      SignalLike,
            RL:      SignalLike,
            R_omega: SignalLike,
            N:       SignalLike,
            dpi:     SignalLike,
            IA:      SignalLike,
            FZ0:     NumberLike,
            R0:      NumberLike,
            V0:      NumberLike,
            Q_FZ1:   NumberLike
    ) -> SignalLike:
        """Returns the vertical load as a function of the free rolling radius and the loaded radius."""

        # unpack tyre properties
        CZ_btm      = self.BOTTOM_STIFF
        R_rim       = self.RIM_RADIUS
        delta_btm   = self.BOTTOM_OFFST

        # asymmetric shift for camber and lateral force (MF 6.2 equation manual)
        ratio = RL / R_omega
        S_FYC = (self.Q_FYS1 + self.Q_FYS2 * ratio + self.Q_FYS3 * ratio ** 2) * IA

        # vertical deflection
        rho_z = self.__find_deflection(R_omega=R_omega, RL=RL, IA=IA)

        # correction term (MF 6.2 equation manual)
        speed_effect    = self.Q_V2 * (R0 / V0) * np.abs(N)
        fx_effect       = ((self.Q_FCX * FX) / FZ0) ** 2
        fy_effect       = ((rho_z / R0) ** self.Q_FCY2 * (self.Q_FCY * (FY - S_FYC) / FZ0)) ** 2
        pressure_effect = 1.0 + self.PFZ1 * dpi
        f_corr = (1.0 + speed_effect - fx_effect - fy_effect) * pressure_effect

        # vertical load
        FZ = f_corr * (Q_FZ1 * (rho_z / R0) + self.Q_FZ2 * (rho_z / R0) ** 2) * FZ0

        # bottoming out force (MF 6.2 equation manual)
        FZ_btm = CZ_btm * (R_rim + delta_btm - RL)

        # total vertical load
        FZ_final = np.maximum(FZ, FZ_btm)
        return FZ_final