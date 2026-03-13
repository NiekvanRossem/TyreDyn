from model_comparison.utils.unit_conversion import radpersec2rpm, si2display
from model_comparison.utils.maps import NAMES_MAP, UNITS_MAP, PARAMETERS, ORDER_TYREDYN, ORDER_MFEVAL
from model_comparison.utils.dataclasses import OutputSignals
from model_comparison.utils.table_print import print_table
from matlab import double as matlab_double
from dataclasses import fields
import numpy as np

def process_tyredyn(output: list):
    """
    Returns the individual channels for the TyreDyn full state output. Units have been converted from SI to display
    units.
    """

    # store output in a dataclass
    tyredyn_signals = list_to_dataclass(output, ORDER_TYREDYN)

    # convert SI to display units
    tyredyn_signals = si2display(tyredyn_signals)

    return tyredyn_signals

def process_mfeval(output: matlab_double):
    """Processes the MFeval output to display units, and stores the results in a dataclass."""

    # store output in a dataclass
    mfeval_signals = list_to_dataclass(output.flatten(), ORDER_MFEVAL)

    # convert SI to display units
    mfeval_signals = si2display(mfeval_signals)

    return mfeval_signals

def plot_comparison(output_tyredyn: list, output_mfeval: matlab_double):
    """Processes the output vectors of both TyreDyn and MFeval, and prints them as a table"""

    # initialize variables
    for param in PARAMETERS:
        globals()[param] = [None, None]

    # process TyreDyn output
    tyredyn_signals = process_tyredyn(output_tyredyn)

    # process MFeval output
    mfeval_signals = process_mfeval(np.array(output_mfeval))

    # print table
    print_table(tyredyn_signals, mfeval_signals)

def list_to_dataclass(output: list, order: list):
    """
    Takes the output of TyreDyn or MFeval and stores the values in the ``OutputSignals`` dataclass. Written by ChatGPT

    Parameters
    ----------
    output : list
        The output of TyreDyn or MFeval
    order : list
        List containing the names of the output signals in the correct order.

    Returns
    -------
    OutputSignals
        The output of TyreDyn or MFeval in a dataclass.
    """

    field_names = {f.name for f in fields(OutputSignals)}
    mapping = dict(zip(order, output))

    output_class = OutputSignals(**{
        k: mapping[k]
        for k in field_names
        if k in order
    })

    return output_class
