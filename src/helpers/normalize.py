from src.utils.formatting import SignalLike
from typing import Union, Literal
import numpy as np

# TODO: rename

class Normalize:
    """
    Module containing the functions to _normalize input signals.
    """

    def __init__(self, model):
        """Make the properties of the overarching ``MF61`` class and other modules available."""
        self._model = model

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def _find_dfz(
            self,
            FZ: SignalLike
    ) -> SignalLike:
        """Finds the normalized vertical load."""

        # unpack parameters
        FZO = self.FNOMIN

        # scale nominal load (4.E1)
        FZO_scaled = self.LFZO * FZO

        # find normalized vertical load (4.E2a)
        dfz = (FZ - FZO_scaled) / FZO_scaled

        return dfz

    def _find_dpi(
            self,
            P: SignalLike
    ) -> SignalLike:
        """Finds the normalized tyre pressure."""

        # extract parameters
        PO = self.NOMPRES

        # normalized pressure (4.E2b)
        dpi = (P - PO) / PO
        return dpi

    def _find_speeds(
            self,
            *,
            SA: SignalLike,
            SL: SignalLike,
            VX: SignalLike
    ) -> SignalLike:
        """Finds the speed of the slip point ``S`` as well as the speed of point ``C``."""

        # longitudinal slip speed (2.11)
        VSX = - SL * VX

        # lateral slip speed (2.12)
        VSY = VX * self.tan(SA)

        # total slip speed (3.39)
        VS = np.sqrt(VSX ** 2 + VSY ** 2)

        # total contact patch speed
        V = np.sqrt(VX ** 2 + VSY ** 2)

        return VS, V

    @staticmethod
    def _replace_value(
            sig_in:     SignalLike,
            *,
            target_sig: SignalLike,
            target_val: Union[float, int],
            new_val:    Union[float, int],
    ) -> SignalLike:
        """
        Replaces the values of ``sig_in`` with ``new_val``, on the indices where ``target_sig`` matches
        ``target_val``.
        """

        # for numpy arrays
        if isinstance(sig_in, np.ndarray):
            sig_in[target_sig == target_val] = new_val

        # for lists
        elif isinstance(sig_in, list):
            sig_in = [new_val if v == target_val else sig_in[v] for v in target_sig]

        # for single values
        else:
            sig_in = new_val if target_sig == target_val else sig_in

        return sig_in

    @staticmethod
    def _flip_negative(sig_in: SignalLike, *, helper_sig) -> SignalLike:
        """Flips the sign of ``sig_in`` on the places where ``helper_sig`` is negative."""

        # find indices where the target signal is negative
        idx = np.where(helper_sig < 0)

        # flip sign of sig
        sig_in[idx] = - sig_in[idx]
        return sig_in

    @staticmethod
    def _correct_signal(
            sig_in: SignalLike,
            *,
            correction_factor: SignalLike,
            helper_sig: SignalLike,
            threshold: NumberLike,
            method: Literal["<", ">", "=", "<=", ">="]
    ) -> SignalLike:
        """Multiplies the input signal with the correction factor, on the indices where the helper signal is less than
        the threshold."""

        if method == "<":
            idx = np.where(helper_sig < threshold)
        elif method == ">":
            idx = np.where(helper_sig > threshold)
        elif method == "=":
            idx = np.where(helper_sig == threshold)
        elif method == "<=":
            idx = np.where(helper_sig <= threshold)
        elif method == ">=":
            idx = np.where(helper_sig >= threshold)
        else:
            raise SyntaxError("method must be <, >, <=, >=")

        sig_in[idx] *= correction_factor[idx]
        return sig_in