import numpy as np

def normalize_fittyp(fittyp: str) -> str:
    """
    Quick function that normalizes the ``FITTYP`` value. Can deal with the following input formats:
      - ``MF 5.2``
      - ``MF5.2``
      - ``MF52``
      - ``5.2``
      - ``52``

    Outputs ``MF``, followed by the model type (e.g. ``5.2`` or ``6.2``), so ``52`` becomes ``MF5.2``.
    Template provided by ChatGPT, and modified from there.
    """

    if not fittyp:
        return "UNKNOWN"

    fittyp = fittyp.strip().upper()
    if fittyp.startswith("MF"):
        fittyp = fittyp.replace("MF", "")
    if "." not in fittyp and len(fittyp) > 1:
        fittyp = f"{fittyp[0]}.{fittyp[1:]}"
    return f"MF{fittyp}"
