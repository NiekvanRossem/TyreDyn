from src.utils.misc import allowableData
import numpy as np

class ContactPatchMF61:
    """
    Contact patch module for MF 6.1.
    """

    def __init__(self, model):
        """Make the properties of the overarching ``MF61`` class and other modules available."""
        self._model = model

        # other modules
        self.stiffness = model.stiffness

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def find_contact_patch(self, FZ: allowableData) -> list[allowableData]:
        """
        Finds the contact patch dimensions.

        :param FZ: vertical load.

        :return: ``a``, ``b`` -- ellipse radii of the contact patch.
        """

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
