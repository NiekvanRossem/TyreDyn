from numpy import pi, rad2deg, deg2rad
from warnings import warn
from model_comparison.utils.dataclasses import OutputSignals
from tyredyn.types.aliases import SignalLike

def radpersec2rpm(sig_in: SignalLike) -> SignalLike:
    """Converts angular speed from rad/s to rpm"""
    return sig_in * 60.0 / (2.0 * pi)

def si2display(sig: OutputSignals):
    """Converts selected signals from SI units to display units."""

    if sig.units == "SI":

        # angular signals rad to degree or from N/rad to N/deg
        sig.SA  = rad2deg(sig.SA)
        sig.IA  = rad2deg(sig.IA)
        sig.KYA = deg2rad(sig.KYA)

        # pressure from Pa to bar
        sig.P  = 1e-5 * sig.P

        # speed from m/s to km/h, and angular speed from rad/s to rpm
        sig.VX = 3.6 * sig.VX
        sig.N = radpersec2rpm(sig.N)

        # stiffness from N/m to N/mm
        sig.Cx = 1e-3 * sig.Cx
        sig.Cy = 1e-3 * sig.Cy
        sig.Cz = 1e-3 * sig.Cz

        # slip stiffness from N/slip to N/0.01slip
        sig.KXK = 1e-2 * sig.KXK

        # lengths from m to mm
        sig.sigma_x = 1e3 * sig.sigma_x
        sig.sigma_y = 1e3 * sig.sigma_y
        sig.RE      = 1e3 * sig.RE
        sig.RL      = 1e3 * sig.RL
        sig.rho     = 1e3 * sig.rho
        sig.t       = 1e3 * sig.t

        # contact patch dimensions converted from m to mm and from diameter to radius
        sig.a = 1e3 * sig.a / 2
        sig.b = 1e3 * sig.b / 2

        sig.units = "display"
    else:
        warn(f"Attempted to convert {sig.__name__} to display units, but input was already in display units. "
             f"Argument passed without modification.")

    return sig
