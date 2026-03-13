from model_comparison.utils.dataclasses import OutputSignals
from model_comparison.utils.maps import NAMES_MAP, UNITS_MAP

def format_value(val, unit, *, value_width, decimals):
    """Returns the formatted value and unit. Written by ChatGPT."""
    return f"{val:>{value_width - len(unit) - 1}.{decimals}f} {unit}"

def print_table_section(
        section_name: str,
        params: dict,
        *,
        section_width: int  = 30,
        value_width: int    = 30,
        diff_width: int     = 14,
        decimals: int       = 3
) -> None:
    """
    Prints the table for one section. Units are automatically mapped based on ``UNITS_MAP``. Template provided by
    ChatGPT, and modified by me.

    Parameters
    ----------
    section_name : str
        Section name.
    params : list
        List of parameters to print
    section_width : int, optional
        Width of the section column, in number of characters.
    value_width : int, optional
        Width of the value columns, in number of characters.
    diff_width : int, optional
        Width of the difference column, in number of characters.
    decimals : int, optional
        Number of decimals to use.
    """

    # print header
    header = (
        f"{section_name:<{section_width}} | "
        f"{'TYREDYN':<{value_width}} | "
        f"{'MFEVAL':<{value_width}} | "
        f"{'DIFFERENCE':<{diff_width}}"
    )
    print(header)
    print("-" * len(header))

    # print variables
    for name in params:

        # extract values and calculate difference
        val_tyredyn = params[name][0]
        val_mfeval  = params[name][1]
        diff   = val_tyredyn - val_mfeval

        # find matching unit
        unit   = UNITS_MAP.get(name, "")

        # find display name
        display_name = NAMES_MAP.get(name, "")

        row = (
            f"{display_name:<{section_width}} | "
            f"{format_value(val_tyredyn, unit, value_width=value_width, decimals=decimals)} | "
            f"{format_value(val_mfeval, unit, value_width=value_width, decimals=decimals)} | "
            f"{diff:>{diff_width}.{decimals}e}"
        )
        print(row)

    # print bottom line
    print("-" * len(header))

def print_table(tyredyn_signals: OutputSignals, mfeval_signals: OutputSignals):
    """Prints a table with both MFeval and TyreDyn outputs."""

    # check if they have been converted to display units
    assert mfeval_signals.units == "display" and tyredyn_signals.units == "display", \
        "Please provide your signals in display units before printing."

    # print table sections
    print('\n')
    print_table_section("Tyre state",
                        {
                            "SA"        : [tyredyn_signals.SA,      mfeval_signals.SA],
                            "SL"        : [tyredyn_signals.SL,      mfeval_signals.SL],
                            "IA"        : [tyredyn_signals.IA,      mfeval_signals.IA],
                            "P"         : [tyredyn_signals.P,       mfeval_signals.P],
                            "PHIT"      : [tyredyn_signals.PHIT,    mfeval_signals.PHIT],
                        })
    print_table_section("Speed",
                        {
                            "VX"        : [tyredyn_signals.VX,      mfeval_signals.VX],
                            "N"         : [tyredyn_signals.N,       mfeval_signals.N],
                        })
    print_table_section("Forces",
                        {
                            "FX"        : [tyredyn_signals.FX,      mfeval_signals.FX],
                            "FY"        : [tyredyn_signals.FY,      mfeval_signals.FY],
                            "FZ"        : [tyredyn_signals.FZ,      mfeval_signals.FZ],
                        })
    print_table_section("Moments",
                        {
                            "MX"        : [tyredyn_signals.MX,      mfeval_signals.MX],
                            "MY"        : [tyredyn_signals.MY,      mfeval_signals.MY],
                            "MZ"        : [tyredyn_signals.MZ,      mfeval_signals.MZ],
                            "MZR"       : [tyredyn_signals.MZR,     mfeval_signals.MZR],
                        })
    print_table_section("Gradients",
                        {
                            "KXK"       : [tyredyn_signals.KXK,     mfeval_signals.KXK],
                            "KYA"       : [tyredyn_signals.KYA,     mfeval_signals.KYA],
                        })
    print_table_section("Friction coefficients",
                        {
                            "mu_x"      : [tyredyn_signals.mu_x,    mfeval_signals.mu_x],
                            "mu_y"      : [tyredyn_signals.mu_y,    mfeval_signals.mu_y],
                        })
    print_table_section("Relaxation lengths",
                        {
                            "sigma_x"   : [tyredyn_signals.sigma_x, mfeval_signals.sigma_x],
                            "sigma_y"   : [tyredyn_signals.sigma_y, mfeval_signals.sigma_y],
                        })
    print_table_section("Radii and deflection",
                        {
                            "RL"        : [tyredyn_signals.RL,      mfeval_signals.RL],
                            "RE"        : [tyredyn_signals.RE,      mfeval_signals.RE],
                            "rho"       : [tyredyn_signals.rho,     mfeval_signals.rho],
                        })
    print_table_section("Stiffness",
                        {
                            "Cx"        : [tyredyn_signals.Cx,      mfeval_signals.Cx],
                            "Cy"        : [tyredyn_signals.Cy,      mfeval_signals.Cy],
                            "Cz"        : [tyredyn_signals.Cz,      mfeval_signals.Cz],
                        })
    print_table_section("Contact patch dimensions",
                        {
                            "a"         : [tyredyn_signals.a,       mfeval_signals.a],
                            "b"         : [tyredyn_signals.b,       mfeval_signals.b],
                            "t"         : [tyredyn_signals.t,       mfeval_signals.t],
                        })
