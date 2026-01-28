from tyredyn.types.aliases import SignalLike
from tyredyn.infrastructure.subsystem_base import SubSystemBase
import numpy as np

class ContactPatchMF6x(SubSystemBase):
    """
    Contact patch module for the MF-Tyre 6.1 and MF-Tyre 6.2 models.
    """

    def _connect(self, model):
        self.stiffness = model.stiffness

    def _find_contact_patch(
            self,
            *,
            FZ: SignalLike,
            P:  SignalLike = None
    ) -> list[SignalLike]:
        """
        Finds the contact patch dimensions.

        Parameters
        ----------
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).

        Returns
        -------
        a, b : list[SignalLike]
            Contact patch length and width.
        """

        # unpack tyre parameters
        R0 = self.UNLOADED_RADIUS
        W  = self.WIDTH

        # vertical stiffness
        CZ = self.stiffness._find_vertical_stiffness(P)

        # length (A3.7)
        a = R0 * (self.Q_RA2 * FZ / (CZ * R0) + self.Q_RA1 * np.sqrt(FZ / (CZ * R0)))

        # half width (A3.8)
        b = W * (self.Q_RB2 * FZ / (CZ * R0) + self.Q_RB1 * (FZ / (CZ * R0)) ** (1/3))

        return [a, b]
