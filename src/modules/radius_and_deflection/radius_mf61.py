from src.utils.misc import allowableData
from typing import Literal
import numpy as np

class RadiusMF61:

    def __init__(self, model):
        """Import the properties of the overarching ``MF61`` class."""
        self._model = model

        # helper functions
        self.normalize  = model.normalize

        # other modules used
        self.stiffness  = model.stiffness

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def find_deflection(
            self,
            FX: allowableData,
            FY: allowableData,
            FZ: allowableData,
            N:  allowableData,
            P:  allowableData = None) -> allowableData:
        """
        Returns the vertical deflection of the tyre. A positive value signifies compression.

        Parameters
        ----------
        FX : allowableData
            Longitudinal force.
        FY : allowableData
            Side force.
        FZ : allowableData
            Vertical load
        N : allowableData
            Angular speed of the wheel.
        P : allowableData, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).

        Returns
        -------
        rho : allowableData
            Vertical deflection.
        """

        # set default values for optional arguments
        P  = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FX, FY, FZ, N, P = self._format_check([FX, FY, FZ, N, P])

        # unpack tyre properties
        R0  = self.UNLOADED_RADIUS
        V0  = self.LONGVL
        FZ0 = self.FNOMIN
        CZ0 = self.VERTICAL_STIFFNESS

        # find QFZ1 from CZ0 (A3.4)
        Q_FZ1 = np.sqrt((CZ0 * R0 / FZ0) ** 2 - 4 * self.Q_FZ2)

        # normalize tyre pressure
        dpi = self.normalize._find_dpi(P)

        # inputs affecting the radius (A3.3) TODO: equation 4.E68 adds extra camber terms to it.
        speed_effect    = self.Q_V2 * np.abs(N) * R0 / V0
        fx_effect       = (self.Q_FCX * FX / FZ0) ** 2
        fy_effect       = (self.Q_FCY * FY / FZ0) ** 2
        pressure_effect = (1.0 + self.PFZ1 * dpi) * FZ0

        # solve via the ABC formula
        A = - self.Q_FZ2 / (R0 ** 2)
        B = - Q_FZ1 / R0
        C = FZ / ((1.0 + speed_effect - fx_effect - fy_effect) * pressure_effect)
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
        rho = np.clip(rho, 1e-6, np.inf)

        # TODO: add bottoming out check

        return rho

    def find_effective_radius(
            self,
            FZ: allowableData,
            N:  allowableData,
            P:  allowableData = None) -> allowableData:
        """
        Returns the effective tyre radius, to be used for calculating the slip ratio.

        Parameters
        ----------
        FZ : allowableData
            Vertical load.
        N : allowableData
            Angular speed of the wheel.
        P : allowableData, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).

        Returns
        -------
        RE : allowableData
            Effective tyre radius.
        """
        
        # set default values for optional arguments
        P  = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, N, P = self._format_check([FZ, N, P])

        # unpack tyre properties
        FZ0 = self.FNOMIN

        # vertical stiffness
        CZ = self.stiffness.find_vertical_stiffness(P)

        # loaded radius
        R_omega = self.find_free_radius(N)

        # effective radius (A3.6) -- FZ trig functions do not get corrected to degrees
        RE = R_omega - FZ0 / CZ * (self.FREFF * FZ / FZ0 + self.DREFF * np.atan2(self.BREFF * FZ / FZ0, 1))
        return RE

    def find_free_radius(self, N: allowableData) -> allowableData:
        """
        Returns the free rolling radius, which captures the tyre growth as it spins up.

        Parameters
        ----------
        N : allowableData
            Angular speed of the wheel.

        Returns
        -------
        R_omega : allowableData
            Free rolling radius.
        """

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            N = self._format_check(N)

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS
        V0 = self.LONGVL

        # free rolling radius (A3.1)
        R_omega = R0 * (self.Q_RE0 + self.Q_V1 * (R0 * N / V0) ** 2)
        return R_omega

    def find_loaded_radius(
            self,
            FX: allowableData,
            FY: allowableData,
            FZ: allowableData,
            N:  allowableData,
            P:  allowableData = None) -> allowableData:
        """
        Returns the loaded radius of the tyre.

        Parameters
        ----------
        FX : allowableData
            Longitudinal force.
        FY : allowableData
            Side force.
        FZ : allowableData
            Vertical load
        N : allowableData
            Angular speed of the wheel.
        P : allowableData, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).

        Returns
        -------
        RL : allowableData
            Loaded radius.
        """

        # set default value for optional argument
        if self._check_format:
            P = self.INFLPRES if P is None else P

        # free radius
        R_omega = self.find_free_radius(N)

        # deflection
        rho = self.find_deflection(FX, FY, FZ, N, P)

        # loaded radius
        RL = R_omega - rho

        return RL
