from src.utils.formatting import SignalLike
import numpy as np

class ContactPatchMF61:
    """
    Contact patch module for MF 6.1.
    """

    def __init__(self, model):
        """Import the properties of the overarching ``MF61`` class."""
        self._model = model

        # other modules
        self.stiffness = model.stiffness

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def find_contact_patch(
            self,
            *,
            FZ: SignalLike,
            P: SignalLike = None
    ) -> list[SignalLike]:
        """
        Finds the contact patch dimensions.

        Parameters
        ----------
        FZ : SignalLike
            Vertical load
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).

        Returns
        -------
        a, b : list[SignalLike]
            Contact patch length and width.
        """

        # set default values for optional arguments
        P = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ = self._format_check(FZ)

        # unpack tyre parameters
        R0 = self.UNLOADED_RADIUS
        W  = self.WIDTH

        # vertical stiffness
        CZ = self.stiffness.find_vertical_stiffness(P)

        # length (A3.7)
        a = R0 * (self.Q_RA2 * FZ / (CZ * R0) + self.Q_RA1 * np.sqrt(FZ / (CZ * R0)))

        # half width (A3.8)
        b = W * (self.Q_RB2 * FZ / (CZ * R0) + self.Q_RB1 * (FZ / (CZ * R0)) ** (1/3))

        return [a, b]
