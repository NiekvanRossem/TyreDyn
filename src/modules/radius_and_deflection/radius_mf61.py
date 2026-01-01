from src.utils.formatting import SignalLike
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
            **kwargs
    ) -> list[SignalLike]:
        """
        Returns the various radii and deflection of the tyre. Order is ``R_omega``, ``RE``, ``RL``, ``rho``.

        Parameters
        ----------
        FZ : SignalLike
            Vertical load.
        N : SignalLike, optional
            Angular speed of the wheel (will be calculated from ``VX`` and ``SL`` if not specified).
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        kwargs : any, optional
            Allows other arguments to be passed for compatibility with ``MF62``. Arguments passed will not be used.

        Returns
        -------
        R_omega, RE, RL, rho : list[SignalLike]
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

        # inputs affecting the radius (A3.3)
        speed_effect    = self.Q_V2 * np.abs(N) * R0 / V0
        fx_effect       = (self.Q_FCX * FX / FZ0) ** 2
        fy_effect       = (self.Q_FCY * FY / FZ0) ** 2
        pressure_effect = (1.0 + self.PFZ1 * dpi)

        # NOTE: equation 4.E68 from the book adds camber dependency to the loaded radius calculation, but MFeval uses
        # A3.3 instead, and we will as well.

        # solve via the ABC formula
        A = - self.Q_FZ2 / (R0 ** 2)
        B = - Q_FZ1 / R0
        C = FZ / ((1.0 + speed_effect - fx_effect - fy_effect) * pressure_effect * FZ0)
        rho = (- B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

        # display warning if only imaginary solutions can be found for a datapoint
        check_root = B ** 2 - 4 * A * C
        if not isinstance(check_root, np.ndarray):
            if isinstance(check_root, list):
                check_root = np.array(check_root)
            else:
                check_root = np.array([check_root])
        if any(check_root < 0.0):
            warnings.warn("No real solution found for the tyre deflection!")

        # apply proper limits to avoid dividing by zero
        rho = np.maximum(rho, 1e-6)

        # loaded radius
        RL = R_omega - rho

        return [R_omega, RE, RL, rho]
