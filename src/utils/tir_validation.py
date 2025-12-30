"""
Filename:         tir_validation.py
Author(s):        Niek van Rossem
Creation date:    31-10-2025

Documentation:
This script contains all the code for the TIR file data validation with Pydantic. Once a TIR file is loaded in, it's
data is stored in a dict. All fields are checked for compatibility with the selected tyre model (e.g. MF6.2). Missing
fields and fields with the wrong data type will raise an error.

To use manually, simply call TIRValidation.validate_with_model(tyre), where `tyre` is your dict with tyre params.

ChatGPT was used for advice regarding best practices and debugging. Code was fully tested and verified before release.

Current version: 1.0

Changelog:
    07-12-2025: First working version. Only supports MF 6.2
"""

from pydantic import BaseModel, RootModel, field_validator, model_validator, PrivateAttr
from typing import Dict, ClassVar
from src.utils.formatting import normalize_fittyp

##--------------------------------------------------------------------------------------------------------------------##

class ModelBootstrap(RootModel[dict[str, float | int | str]]):
    """
    Checks the MODEL section of the TIR file beforehand to make the FITTYP parameter available. Written by ChatGPT.
    """
    pass

class DictSection:
    """
    Base class that is used to validate the parameters in each header of the tyre dictionary. FloatSection and
    UnitSection are subclasses that inherit this class.
    """

    # initialize dictionary with required parameters
    required_params: ClassVar[dict[str, set[str]]] = {}

    @classmethod
    def validate_dict(cls, v: Dict[str, object], info) -> Dict[str, object]:

        # get current tyre model
        model = info.context.get("fittyp", "UNKNOWN")
        if model == "UNKNOWN":
            raise NameError("Cannot obtain the current model type")

        # find required parameters for current tyre model
        required = cls.required_params.get("Any", set()) | cls.required_params.get(model, set())

        # check for any missing parameters
        missing = required - v.keys()
        if missing:
            raise ValueError(f"{cls.__name__}: missing parameters for {model}: {', '.join(missing)}")
        return v

class FloatSection(DictSection, RootModel[Dict[str, float | int]]):
    """Subclass to validate all TIR file sections that contain exclusively floats."""
    @field_validator('root', check_fields=False)
    def validate_root(cls, v, info):
        return cls.validate_dict(v, info)

##--------------------------------------------------------------------------------------------------------------------##

# validation class for UNITS section
class UnitsSection(DictSection, RootModel[Dict[str, str]]):
    """Subclass to validate the UNITS section of a TIR file."""

    # initialize model type field and dictionary with required parameters
    required_params: ClassVar[dict[str, str]] = {
        "Any": {
            "LENGTH", "FORCE", "ANGLE", "MASS", "TIME"
        }
    }

    @field_validator('root', check_fields=False)
    def validate_root(cls, v, info):
        return cls.validate_dict(v, info)

# validation class for MODEL section
class ModelSection(RootModel[Dict[str, float | int | str]]):
    """Subclass to validate the MODEL section of a TIR file."""

    # initialize dictionary with required parameters
    required_params: ClassVar[dict[str, set[str]]] = {
        "Any": {
            "FITTYP", "TYRESIDE", "LONGVL", "VXLOW", "ROAD_INCREMENT", "ROAD_DIRECTION"
        }
    }

    # check if all required parameters are present (updated version to account for the various data types
    @field_validator('root')
    def validate_model(cls, v, info):

        # get current tyre model
        model = info.context.get("fittyp", "UNKNOWN")
        if model == "UNKNOWN":
            raise NameError("Cannot obtain the current model type")

        # find required parameters for current tyre model
        required = cls.required_params.get(model, set())

        # check for any missing parameters
        missing = required - v.keys()
        if missing:
            raise ValueError(f"{cls.__name__} Missing parameters for {model}: {','.join(missing)}")

        # TYRESIDE should be a string. FITTYP may be a string. All others must be a float or int
        for key, val in v.items():
            if key == "TYRESIDE" and not isinstance(val, str):
                raise TypeError("TYRESIDE must be a string! (either 'LEFT' or 'RIGHT')")
            if key == "FITTYP" and not isinstance(val, str | int):
                raise TypeError("FITTYP must be a either be a string or an int!")
            elif key != "TYRESIDE" and key != "FITTYP" and not isinstance(val, (float, int)):
                raise TypeError(f"{key} must be a float or int!")
        return v

# validation class for DIMENSION section
class DimensionSection(FloatSection):
    """Subclass to validate the DIMENSION section of a TIR file."""

    required_params = {
        "Any": {
            "UNLOADED_RADIUS", "WIDTH", "RIM_RADIUS", "RIM_WIDTH", "ASPECT_RATIO"
        }
    }

##--------------------------------------------------------------------------------------------------------------------##

"""
The following classes are used to validate the parameters in all the TIR file sections containing only floats. 

To add support for a new tyre model, simply add its name as a header to the `required_params` dictionary, and add all 
the params that should be present to this section. Currently only supports tyre tyre_models formatted as MFx.x_tyre. Other names 
will either be reformatted to MFx.x_tyre if they're close enough, or an error will occur. See `normalize_fittyp` for more 
information.  

To add a new parameter category, create a new class that inherits `FloatSection`, and define `required_parameters` 
inside it as a dictionary.
"""

class OperatingConditionsSection(FloatSection):
    required_params = {
        "Any": {"INFLPRES", "NOMPRES"
        }
    }

class InertiaSection(FloatSection):
    required_params = {
        "Any": {"MASS", "IXX", "IYY", "BELT_MASS", "BELT_IXX", "BELT_IYY", "GRAVITY"
        }
    }

class VerticalSection(FloatSection):
    required_params = {
        "MF6.2": {
            "FNOMIN", "VERTICAL_STIFFNESS", "VERTICAL_DAMPING", "MC_CONTOUR_A", "MC_CONTOUR_B", "BREFF", "DREFF",
            "FREFF", "Q_RE0", "Q_V1", "Q_V2", "Q_FZ2", "Q_FCX", "Q_FCY", "Q_CAM", "PFZ1", "Q_FCY2", "Q_CAM1", "Q_CAM2",
            "Q_CAM3", "Q_FYS1", "Q_FYS2", "Q_FYS3", "BOTTOM_OFFST", "BOTTOM_STIFF"
        }
    }

class StructuralSection(FloatSection):
    required_params = {
        "MF6.2": {
            "LONGITUDINAL_STIFFNESS", "LATERAL_STIFFNESS", "YAW_STIFFNESS", "FREQ_LONG", "FREQ_LAT", "FREQ_YAW",
            "FREQ_WINDUP", "DAMP_LONG", "DAMP_LAT", "DAMP_YAW", "DAMP_WINDUP", "DAMP_RESIDUAL", "DAMP_VLOW", "Q_BVX",
            "Q_BVT", "PCFX1", "PCFX2", "PCFX3", "PCFY1", "PCFY2", "PCFY3", "PCMZ1"
        }
    }

class ContactPatchSection(FloatSection):
    required_params = {
        "MF6.2": {
            "Q_RA1", "Q_RA2", "Q_RB1", "Q_RB2", "ELLIPS_SHIFT", "ELLIPS_LENGTH", "ELLIPS_HEIGHT", "ELLIPS_ORDER",
            "ELLIPS_MAX_STEP", "ELLIPS_NWIDTH", "ELLIPS_NLENGTH", "ENV_C1", "ENV_C2"
        }
    }

class LongitudinalSection(FloatSection):
    required_params = {
        "MF6.2": {
            "PCX1", "PDX1", "PDX2", "PDX3", "PEX1", "PEX2", "PEX3", "PEX4", "PKX1", "PKX2", "PKX3", "PHX1", "PHX2",
            "PVX1", "PVX2", "PPX1", "PPX2", "PPX3", "PPX4", "RBX1", "RBX2", "RBX3", "RCX1", "REX1", "REX2", "RHX1"
        }
    }

class InflationSection(FloatSection):
    required_params = {
        "Any": {
            "PRESMIN",
            "PRESMAX"
        }
    }

class VerticalForceSection(FloatSection):
    required_params = {
        "Any": {
            "FZMIN",
            "FZMAX"
        }
    }

class LongSlipSection(FloatSection):
    required_params = {
        "Any": {
            "KPUMIN",
            "KPUMAX"
        }
    }

class SlipAngleSection(FloatSection):
    required_params = {
        "Any": {
            "ALPMIN",
            "ALPMAX"
        }
    }

class InclinationAngleSection(FloatSection):
    required_params = {
        "Any": {
            "CAMMIN",
            "CAMMAX"
        }
    }

class ScalingSection(FloatSection):
    required_params = {
        "MF6.2": {
            "LFZO", "LCX", "LMUX", "LEX", "LKX", "LHX", "LVX", "LCY", "LMUY", "LEY", "LKY", "LKYC", "LKZC", "LHY",
            "LVY", "LTR", "LRES", "LXAL", "LYKA", "LVYKA", "LS", "LMX", "LVMX", "LMY", "LMP"
        }
    }

class OverturningSection(FloatSection):
    required_params = {
        "MF6.2": {
            "QSX1", "QSX2", "QSX3", "QSX4", "QSX5", "QSX6", "QSX7", "QSX8",
            "QSX9", "QSX10", "QSX11", "QSX12", "QSX13", "QSX14", "PPMX1"
        }
    }

class LateralSection(FloatSection):
    required_params = {
        "MF6.2": {
            "PCY1", "PDY1", "PDY2", "PDY3", "PEY1", "PEY2", "PEY3", "PEY4", "PEY5", "PKY1", "PKY2", "PKY3", "PKY4",
            "PKY5", "PKY6", "PKY7", "PHY1", "PHY2", "PVY1", "PVY2", "PVY3", "PVY4", "PPY1", "PPY2", "PPY3", "PPY4",
            "PPY5", "RBY1", "RBY2", "RBY3", "RBY4", "RCY1", "REY1", "REY2", "RHY1", "RHY2", "RVY1", "RVY2", "RVY3",
            "RVY4", "RVY5", "RVY6"
        }
    }

class RollingSection(FloatSection):
    required_params = {
        "MF6.2": {
            "QSY1", "QSY2", "QSY3", "QSY4", "QSY5", "QSY6", "QSY7", "QSY8"
        }
    }

class AligningSection(FloatSection):
    required_params = {
        "MF6.2": {
            "QBZ1", "QBZ2", "QBZ3", "QBZ4", "QBZ5", "QBZ9", "QBZ10", "QCZ1", "QDZ1", "QDZ2", "QDZ3", "QDZ4", "QDZ6",
            "QDZ7", "QDZ8", "QDZ9", "QDZ10", "QDZ11", "QEZ1", "QEZ2", "QEZ3", "QEZ4", "QEZ5", "QHZ1", "QHZ2", "QHZ3",
            "QHZ4", "PPZ1", "PPZ2", "SSZ1", "SSZ2", "SSZ3", "SSZ4"
        }
    }

class TurnSlipSection(FloatSection):
    required_params = {
        "MF6.2": {
            "PDXP1", "PDXP2", "PDXP3", "PKYP1", "PDYP1", "PDYP2", "PDYP3", "PDYP4", "PHYP1", "PHYP2",
            "PHYP3", "PHYP4", "PECP1", "PECP2", "QDTP1", "QCRP1", "QCRP2", "QBRP1", "QDRP1"
        }
    }

##--------------------------------------------------------------------------------------------------------------------##

# main validator class for the TIR file data
class TIRValidation(BaseModel):
    """
    Main class for handling TIR validation. Takes a dictionary with tyre params as an input, and checks whether all
    params are present and have the correct data type. To use this class, call the function
    `TIRValidation.validate_with_model(tyre)`, where `tyre` is a dictionary containing the tyre parameters.
    """

    # check all the data
    UNITS:                      UnitsSection
    MODEL:                      ModelSection
    DIMENSION:                  DimensionSection
    OPERATING_CONDITIONS:       OperatingConditionsSection
    INERTIA:                    InertiaSection
    VERTICAL:                   VerticalSection
    STRUCTURAL:                 StructuralSection
    CONTACT_PATCH:              ContactPatchSection
    INFLATION_PRESSURE_RANGE:   InflationSection
    VERTICAL_FORCE_RANGE:       VerticalForceSection
    LONG_SLIP_RANGE:            LongSlipSection
    SLIP_ANGLE_RANGE:           SlipAngleSection
    INCLINATION_ANGLE_RANGE:    InclinationAngleSection
    SCALING_COEFFICIENTS:       ScalingSection
    LONGITUDINAL_COEFFICIENTS:  LongitudinalSection
    OVERTURNING_COEFFICIENTS:   OverturningSection
    LATERAL_COEFFICIENTS:       LateralSection
    ROLLING_COEFFICIENTS:       RollingSection
    ALIGNING_COEFFICIENTS:      AligningSection
    TURNSLIP_COEFFICIENTS:      TurnSlipSection

    @classmethod
    def validate_with_model(cls, data: dict):
        """
        Call this function to validate the data from the tyre model. Written by ChatGPT.

        :param data: dictionary containing the TIR file parameters.
        :return: validated data.
        """

        bootstrap = ModelBootstrap.model_validate(data["MODEL"])
        fittyp = normalize_fittyp(str(bootstrap.root.get("FITTYP", "UNKNOWN")))
        validated = cls.model_validate(data, context = {"fittyp": fittyp})

        print(f"Validation of TIR file successful. All parameters found for version {fittyp}")
        return validated

"""
def validate_data(params):
    \"""
    Validate the TIR file.

    :param params: Dictionary of parameter names and values.
    \"""
    TIRValidation.validate_with_model(params)
"""