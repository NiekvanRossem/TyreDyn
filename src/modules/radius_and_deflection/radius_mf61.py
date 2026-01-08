from src.utils.formatting import SignalLike, NumberLike
from typing import Literal
import numpy as np
import warnings

class RadiusMF61:
    """
    Radius and deflection module for the MF 6.1 tyre model.
    """

    def __init__(self, model):
        """Import the properties of the overarching ``MF61`` class."""
        self._model = model

        # helper functions
        self.normalize  = model.normalize
        self.common = model.common

        # other modules used
        self.stiffness  = model.stiffness

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def _find_radius(
            self,
            *,
            FX: SignalLike,
            FY: SignalLike,
            FZ: SignalLike,
            N:  SignalLike,
            P:  SignalLike = None,
            maxiter: int   = 30,
            tolx: float    = 1e-6
    ) -> list[SignalLike]:
        """
        Returns the various radii and deflection of the tyre. Order is ``R_omega``, ``RE``, ``RL``, ``rho_z``.

        Parameters
        ----------
        maxiter
        tolx
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
        R_omega = self.common._find_free_radius(N=N, R0=R0, V0=V0)

        # effective radius
        RE = self.common._find_effective_radius(FZ=FZ, P=P, R_omega=R_omega, FZ0=FZ0)

        # find QFZ1 from CZ0 (A3.4)
        Q_FZ1 = np.sqrt((CZ0 * R0 / FZ0) ** 2 - 4 * self.Q_FZ2)

        # set bounds for optimization
        RL_lower  = 0.95 * R_omega
        RL_upper  = R_omega
        RL_newest = None

        # evaluate function for initial bounds
        y_lower = self.__find_fz(FX=FX, FY=FY, RL=RL_lower, R_omega=R_omega, N=N, dpi=dpi, FZ0=FZ0, R0=R0, V0=V0, Q_FZ1=Q_FZ1) - FZ
        y_upper = self.__find_fz(FX=FX, FY=FY, RL=RL_upper, R_omega=R_omega, N=N, dpi=dpi, FZ0=FZ0, R0=R0, V0=V0, Q_FZ1=Q_FZ1) - FZ

        # counter TODO: add convergence flags
        counter = 0

        # perform optimization
        for iteration in range(maxiter):

            # make new guess for RL
            RL_newest = RL_upper - (RL_upper - RL_lower) / (y_upper - y_lower) * y_upper
            y_newest = self.__find_fz(FX=FX, FY=FY, RL=RL_newest, R_omega=R_omega, N=N, dpi=dpi, FZ0=FZ0, R0=R0,
                                      V0=V0, Q_FZ1=Q_FZ1) - FZ

            # update values
            y_upper = y_newest
            RL_upper = RL_newest

            # check if all values have converged
            error = abs(y_newest).max()
            if error < tolx * FZ0:
                break
            if counter == maxiter - 1:
                warnings.warn(
                    f"Maximum number of iterations reached. No solution for the loaded radius found. Final error is {error:.6e}")
            counter += 1
        RL = RL_newest

        # vertical deflection
        rho_z = np.maximum(R_omega - RL, 1e-12)

        # inputs affecting the radius (A3.3)
        #speed_effect    = self.Q_V2 * np.abs(N) * R0 / V0
        #fx_effect       = (self.Q_FCX * FX / FZ0) ** 2
        #fy_effect       = (self.Q_FCY * FY / FZ0) ** 2
        #pressure_effect = (1.0 + self.PFZ1 * dpi)

        # NOTE: equation 4.E68 from the book adds camber dependency to the loaded radius calculation, but MFeval uses
        # A3.3 (also in the book) instead, since Q_FZ3 is not a standard parameter, and no equation to calculate this is
        # provided.

        # solve via the ABC formula
        #A = - self.Q_FZ2 / (R0 ** 2)
        #B = - Q_FZ1 / R0
        #C = FZ / ((1.0 + speed_effect - fx_effect - fy_effect) * pressure_effect * FZ0)
        #rho_z = (- B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

        # display warning if only imaginary solutions can be found for a datapoint
        #check_root = B ** 2 - 4 * A * C
        #if not isinstance(check_root, np.ndarray):
        #    if isinstance(check_root, list):
        #        check_root = np.array(check_root)
        #    else:
        #        check_root = np.array([check_root])
        #if any(check_root < 0.0):
        #    warnings.warn("No real solution found for the tyre deflection!")

        # apply proper limits to avoid dividing by zero
        #rho_z = np.maximum(rho_z, 1e-6)

        # loaded radius
        #RL = R_omega - rho_z
        return [R_omega, RE, RL, rho_z]

    def __find_fz(
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
            Q_FZ1:   NumberLike
    ) -> SignalLike:
        """
        Returns the vertical load as a function of the free rolling radius and the loaded radius.

        Parameters
        ----------
        FX
        FY
        RL
        R_omega
        N
        dpi
        FZ0
        R0
        V0
        Q_FZ1

        Returns
        -------

        """

        # unpack tyre properties
        CZ_btm    = self.BOTTOM_STIFF
        R_rim     = self.RIM_RADIUS
        delta_btm = self.BOTTOM_OFFST

        # vertical deflection
        rho_z = R_omega - RL

        # vertical load (A3.3)
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

        # bottoming out force (MF 6.2 equation manual)
        FZ_btm = CZ_btm * (R_rim + delta_btm - RL)

        # total vertical load
        FZ_final = np.maximum(FZ, FZ_btm)
        return FZ_final
