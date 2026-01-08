from src.utils.paths import TYRE_DIR


def read_tir(filename: str) -> dict:
    """
    Reads the TIR file, and store parameters as a dictionary.

    :param filename: Name of the TIR file to be read. Will assume this file is stored inside `example_tyres/tyres`.
    :return: Dictionary of parameter names and values.
    """

    # code for reading a TIR file. Outputs a dictionary with the tyre params.
    with open(TYRE_DIR / filename) as f:
        data = f.readlines()
        params = {}
        paramslist = []

        # loop over all the lines
        for line in data:
            line = line.strip()

            # non-example_tyres line
            if line.startswith('$-') or line.startswith('!'):
                continue

            # start of a new section (create dict entry for it)
            if line.startswith('[') and line.endswith(']'):
                current_header = line[1:-1]
                params[current_header] = {}

            # line contains useful example_tyres
            elif '=' in line:

                # filter out the comment if there is one
                if '$' in line and not line.startswith('$'):
                    line, _ = line.split('$', 1)

                # add to dictionary
                key, value = line.split('=')
                params[current_header][key.strip()] = value.strip()
                paramslist.append(key.strip())

        # convert all the numbers to floats
        for header in params.keys():
            for key in params[header].keys():
                try:
                    if key == "FITTYP":
                        continue
                    else:
                        params[header][key] = float(params[header][key])

                except ValueError:
                    continue
    return params
