from typing import Union
import numpy as np

# create alias for allowable data types
numberData = Union[int, float]
allowableData = Union[numberData, list[numberData], tuple[numberData], np.typing.NDArray[numberData]]

"""
def check_format(data: Union[allowableData, list[allowableData]]) -> allowableData:
    \"""
    Checks the shape of the input data, and flattens them to 1D arrays if needed.

    :param data: input array.
    :return: ``data`` -- flattened into 1D array.
    \"""

    # if a list of channels is passed
    if isinstance(data, list):
        for i, channel in enumerate(data):
            if isinstance(channel, np.ndarray):
                if channel.ndim == 2:
                    assert channel.shape[1] == 1, "Please input a 1D array."
                    data[i] = channel.flatten()

    # if just a single channel is passed
    else:
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                assert data.shape[1] == 1, "Please input a 1D array."
                data = data.flatten()
    return data
"""

def normalize_fittyp(fittyp: str) -> str:
    """
    Quick function that normalizes the FITTYP value. Can deal with the following input formats:
    - MF 5.2
    - MF5.2
    - MF52
    - 5.2
    - 52

    Outputs MF, followed by the model type (e.g. 5.2 or 6.2), so `52` becomes `MF5.2`.
    Written by ChatGPT.
    """

    if not fittyp:
        return "UNKNOWN"

    fittyp = fittyp.strip().upper()
    if fittyp.startswith("MF"):
        fittyp = fittyp.replace("MF", "")
    if "." not in fittyp and len(fittyp) > 1:
        fittyp = f"{fittyp[0]}.{fittyp[1:]}"
    return f"MF{fittyp}"
