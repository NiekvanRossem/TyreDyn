from src.utils.model_map import MODEL_CLASS_MAP
from src.utils.read_file import read_tir
from src.utils.tir_validation import validate_data
from src.utils.misc import *

class Tyre:
    """
    Initialization class for tyre tyre_models. This class contains the functions to read and validate a TIR file. Not
    to be confused with ``TyreBase``, which contains the template for the subclasses.

    The following user-defined settings can be specified:
      - ``check_format`` -- checks the shape of the input arrays, and flattens them if needed (default is ``True``).
      - ``check_limits`` -- checks if all values in the input array are within the specified limits (default is ``True``).
      - ``print_status`` -- set to `False` if you don't want to print status messages after loading (default is `True`).
      - ``use_alpha_star`` -- slip angle correction for large angles and reverse running (default is ``True``).
      - ``use_gamma_star`` -- inclination angle correction for large angles (default is ``True``).
      - ``use_lmu_star`` -- composite friction scaling factor with slip speed (default is ``True``).

    Currently supported tyre models:
      - MF 6.1.2

    :param filename: filename of the TIR file
    :param validate: Set to True to validate the TIR file

    :return: An instance of the chosen tyre class, which can return the state of the tyre for a given input.
    """

    def __new__(cls, filename: str, validate: bool = True, **settings):

        # option to disable printing messages after loading in
        print_status: bool = settings.get('print_status', True)

        # TODO: add support for custom paths
        # TODO: rename filename to filepath
        # TODO: figure out the best way to have the user select a default folder (make this part of initialize procedure)
        # 1. split filepath up into path and filename
        # 2. if only a filename is specified, try searching in the data folder
        # 3. if a filepath is specified, try to find the file there
        # 3. display an error message if file cannot be found

        # read TIR file
        params = read_tir(filename)
        model_type = normalize_fittyp(params.get("MODEL", {}).get("FITTYP", None))
        if print_status:
            print(f"TIR file '{filename}' successfully loaded.")
        if validate:
            validate_data(params)

        # map the FITTYP parameter to the right subclass
        try:
            subclass = MODEL_CLASS_MAP[model_type]
        except KeyError:
            raise KeyError(f"Model type '{model_type}' not recognized. Cannot create the desired instance.")

        # create instance of the new subclass
        obj = super().__new__(subclass)
        obj.__init_from_data__(params, **settings)
        if print_status:
            print(f"Tyre instance of type {subclass} successfully created.")
        return obj

    def __init__(self, filename, validate: bool = True, **settings):
        pass


#----------------------------------------------------------------------------------------------------------------------#
# test script

if __name__ == "__main__":
    current_tyre = Tyre('car205_60R19.tir', validate=True, use_alpha_star=False, check_limits=False)

    FZ = 600
    SL = 0.1
    FX = current_tyre.find_fx_pure(SL, FZ, angle_unit='deg')
    RL = current_tyre.find_loaded_radius(FX, 0.0, FZ, 20.0)
    print(f"== TEST OUTPUT == \n"
          f"vertical load:  {FZ} N \n"
          f"slip ratio:     {SL} \n"
          f"tractive force: {FX:.2f} N \n"
          f"loaded_radius:  {RL:.2f} m")