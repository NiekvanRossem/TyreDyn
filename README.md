# TyreDyn
TyreDyn is an open source tyre modelling package for Python. The aim is to provide a standard python interface for the various tyre models in existence, similar to what [MFeval](https://mfeval.wordpress.com/) is in MATLAB.

## Current status
This package is currently a work in progress. The current status (as of 29 December 2025) is:
- MF 6.1 and MF 6.2 are implemented and functional
- Still needs detailed testing to check whether `TyreDyn`, `MFeval`, and `MF-Tool` give the same outputs.

## How to use
Load in your TIR file by calling `current_tyre = Tyre(<filename>.tir)`. This will create an instance of the tyre model class corresponding to the `FITTYP` parameter.

## Currently supported tyre models
- MF 6.1
- MF 6.2

## Dependencies:
- NumPy
- Pydantic (TIR validation only)
- Matplotlib (example notebooks only)

## Contributors
- Niek van Rossem

## References
1. Pacejka, H.B. & Besselink, I. (2012). *Tire and Vehicle Dynamics. Third Edition*. Elsevier.
   [doi: 10.1016/c2010-0-68548-8](https://doi.org/10.1016/c2010-0-68548-8)
2. Marco Furlan (2025). *MFeval*. MATLAB Central File Exchange. Retrieved December 18, 2025.
   [mathworks.com/matlabcentral/fileexchange/63618-mfeval](mathworks.com/matlabcentral/fileexchange/63618-mfeval)
3. Netherlands Organization for Applied Scientific Research (2013). *MF-Tyre/MF-Swift 6.2 -- Equation manual*.
   Document revision 20130706.
4. International Organization for Standardization (2011). *Road vehicles -- Vehicle dynamics and road-holding
   ability -- Vocabulary* (ISO standard No. 8855:2011)
   [iso.org/obp/ui/#iso:std:iso:8855:ed-2:v1:en](https://www.iso.org/obp/ui/#iso:std:iso:8855:ed-2:v1:en)
5. LeMesurier, B. & Roberts, S. (2025). *Numerical Methods and Analysis with Python*. College of Charleston. Retrieved 28 December 2025.
   [https://lemesurierb.people.charleston.edu/numerical-methods-and-analysis-python/index.html](https://lemesurierb.people.charleston.edu/numerical-methods-and-analysis-python/index.html)
