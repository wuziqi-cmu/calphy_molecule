"""
calphy: a Python library and command line interface for automated free
energy calculations.

Copyright 2021  (c) Sarath Menon^1, Yury Lysogorskiy^2, Ralf Drautz^2
^1: Max Planck Institut für Eisenforschung, Dusseldorf, Germany 
^2: Ruhr-University Bochum, Bochum, Germany

calphy is published and distributed under the Academic Software License v1.0 (ASL). 
calphy is distributed in the hope that it will be useful for non-commercial academic research, 
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
See the ASL for more details. 

More information about the program can be found in:
Menon, Sarath, Yury Lysogorskiy, Jutta Rogal, and Ralf Drautz.
“Automated Free Energy Calculation from Atomistic Simulations.” Physical Review Materials 5(10), 2021
DOI: 10.1103/PhysRevMaterials.5.103801

For more information contact:
sarath.menon@ruhr-uni-bochum.de/yury.lysogorskiy@icams.rub.de
"""

import os
import yaml
import warnings

def check_and_convert_to_list(data):
    """
    Check if the given item is a list, if not convert to a single item list

    Parameters
    ----------
    data : single value or list

    Returns
    -------
    data : list
    """
    if not isinstance(data, list):
        return [data]
    else:
        return data

def fix_paths(potlist): 
    """
    Fix paths for potential files to complete ones
    """
    fixedpots = []
    for pot in potlist:
        pcraw = pot.split()
        if len(pcraw) >= 3 and os.path.exists(pcraw[2]):
            filename = pcraw[2]
            filename = os.path.abspath(filename)
            pcnew = " ".join([*pcraw[:2], filename, *pcraw[3:]])
            fixedpots.append(pcnew)
        else:
            fixedpots.append(pcraw)
        #print(pcnew)
    return fixedpots
    
def prepare_optional_keys(calc, cdict):

    #optional keys
    keydict = {
        "repeat": [1, 1, 1],
        "nsims": 1,
        "thigh": 2.0*cdict["temperature_stop"],
        "npt": True,
        "tguess": None,
        "dtemp": 200,
        "maxattempts": 5,
    }

    for key, val in keydict.items():
        if key in calc.keys():
            cdict[key] = calc[key]
        else:
            cdict[key] = val

    if not (cdict["repeat"][0] == cdict["repeat"][1] == cdict["repeat"][2]):
        raise ValueError("For LAMMPS structure creation, use nx=ny=nz")

    return cdict

def read_yamlfile(file):
    """
    Read a yaml input file
    Parameters
    ----------
    inputfile: string
        name of inout yaml file
    Returns
    -------
    dict: dict
        the read input dict options
    """
    #there are three blocks - main, md and queue
    #main block has subblocks of calculations

    #we need to set up def options
    options = {}
    
    #main dictionary
    options["element"]: None
    options["mass"]: 1.00
    options["units"] = "metal"
    options["atom_style"] = "atomic"

    #create a list for calculations
    options["calculations"] = []

    #options for md
    options["md"] = {
        #pair elements
        "pair_style": None,
        "pair_coeff": None,
        "kspace_style": None,
        "bond_style": None,
        "angle_style": None,
        "dihedral_style": None,
        "improper_style": None,
        "bond_coeff": None,
        "angle_coeff": None,
        "special_bonds": None,
        "neighbor": None,
        "neigh_modify": None,
        #time related properties
        "timestep": 0.001,
        "nsmall": 10000,
        "nevery": 10,
        "nrepeat": 10,
        "ncycles": 100,
        #ensemble properties
        "tdamp": 0.1,
        "pdamp": 0.1,
        #eqbr and switching time
        "te": 25000,
        "ts": 50000,
        #enable separate switching time for rs
        "ts_rs": 50000,
        "tguess": None,
        "dtemp": 200,
        "maxattempts": 5,
        "traj_interval": 0,
    }

    #queue properties
    options["queue"] = {
        "scheduler": "local",
        "cores": 1,
        "jobname": "ti",
        "walltime": "23:50:00",
        "queuename": None,
        "memory": "3GB",
        "commands": None,
        "modules": None,
        "options": None
    }

    #convergence factors that can be set if required
    options["conv"] = {
        "alat_tol": 0.0002,
        "k_tol": 0.01,
        "solid_frac": 0.7,
        "liquid_frac": 0.05,
        "p_tol": 0.5,
    }

    #keys that need to be read in directly
    directkeys = ["md", "queue", "conv"]

    #now read the file
    if os.path.exists(file):
        with open(file) as file:
            indata = yaml.load(file, Loader=yaml.FullLoader)
    else:
        raise FileNotFoundError('%s input file not found'% file)


    #now read keys
    for okey in directkeys:
        if okey in indata.keys():
            for key, val in indata[okey].items():
                options[okey][key] = indata[okey][key] 

    options["element"] = check_and_convert_to_list(indata["element"])
    options["mass"] = check_and_convert_to_list(indata["mass"])
    options["units"] = indata["units"]
    options["atom_style"] = indata["atom_style"]
    # options["md"]["pair_style"] = check_and_convert_to_list(indata["md"]["pair_style"])
    options["md"]["pair_style"] = indata["md"]["pair_style"]
    toBeFixed = ["pair_coeff", "kspace_style", "bond_coeff", "angle_coeff", "special_bonds", "neighbor", "neigh_modify"]
    # is this way of fixing coeff right?
    # for fixing in toBeFixed:
    #     options["md"][fixing] = fix_paths(check_and_convert_to_list(indata["md"][fixing]))

    #now modify ts;
    if isinstance(options["md"]["ts"], list):
        ts1 = options["md"]["ts"][0]
        ts2 = options["md"]["ts"][1]
        options["md"]["ts"] = ts1
        options["md"]["ts_rs"] = ts2
    else:
        options["md"]["ts_rs"] = options["md"]["ts"]

    if not len(options["element"]) == len(options["mass"]):
        raise ValueError("length of elements and mass should be same!")
    options["nelements"] = len(options["element"])

    #now we need to process calculation keys
    #loop over calculations
    if "calculations" in indata.keys():
        #if the key is present
        #Loop 0: over each calc block
        #Loop 1: over lattice
        #Loop 2: over pressure
        #Loop 3: over temperature if needed - depends on mode
        for calc in indata["calculations"]:
            #check and convert items to lists if needed
            mode = calc["mode"]
            
            #now start looping
            #First handle the complex protocols, otherwise go to other simple protocols
            if 'lattice' in calc.keys():
                lattice = check_and_convert_to_list(calc["lattice"])
            else:
                lattice = []
            if 'pressure' in calc.keys():
                pressure = check_and_convert_to_list(calc["pressure"])
            else:
                pressure = []


            if (mode=='melting_temperature'):
                cdict = {}
                cdict["mode"] = calc["mode"]
                cdict["temperature"] = 0
                cdict["temperature_stop"] = 0

                #now if lattice is provided-> use that; but length should be one
                if len(lattice) == 1:
                    cdict["lattice"] = lattice[0]
                elif len(lattice)>1:
                    raise ValueError('For melting_temperature mode, please provide only one lattice')
                else:
                    cdict["lattice"] = None

                if len(pressure) == 1:
                    cdict["pressure"] = pressure[0]
                elif len(pressure)>1:
                    raise ValueError('For melting_temperature mode, please provide only one pressure')
                else:
                    cdict["pressure"] = 0

                #pressure is zero                
                cdict["state"] = None
                cdict["nelements"] = options["nelements"]
                cdict["element"] = options["element"]
                cdict["lattice_constant"] = 0
                cdict["iso"] = False
                cdict["fix_lattice"] = False
                cdict = prepare_optional_keys(calc, cdict)
                options["calculations"].append(cdict)                      


            #now handle other normal modes
            else:   
                state = check_and_convert_to_list(calc["state"])
                temperature = check_and_convert_to_list(calc["temperature"])

                #prepare lattice constant values
                if "lattice_constant" in calc.keys():
                    lattice_constant = check_and_convert_to_list(calc["lattice_constant"])
                else:
                    lattice_constant = [0 for x in range(len(lattice))]
                #prepare lattice constant values
                if "baro" in calc.keys():
                    baro = check_and_convert_to_list(calc["baro"])
                else:
                    baro = [False for x in range(len(lattice))]

                if "fix_lattice" in calc.keys():
                    fix_lattice = check_and_convert_to_list(calc["fix_lattice"])
                else:
                    fix_lattice = [False for x in range(len(lattice))]


                for i, lat in enumerate(lattice):
                    if (mode == "ts") or (mode == "mts") or (mode == "tscale"):
                        for press in pressure:
                            cdict = {}
                            cdict["mode"] = calc["mode"]
                            #we need to check for temperature length here
                            if not len(temperature)==2:
                                raise ValueError("At least two temperature values are needed for ts/tscale")
                            cdict["temperature"] = temperature[0]
                            cdict["pressure"] = press
                            cdict["lattice"] = lat
                            if state[i] in ['solid', 'liquid']:
                                cdict["state"] = state[i]
                            else:
                                raise ValueError('state has to be either solid or liquid')
                            cdict["temperature_stop"] = temperature[-1]
                            cdict["nelements"] = options["nelements"]
                            cdict["element"] = options["element"]
                            cdict["lattice_constant"] = lattice_constant[i]
                            # cdict["iso"] = iso[i]
                            cdict["baro"] = baro[i]
                            cdict["fix_lattice"] = fix_lattice[i]
                            cdict = prepare_optional_keys(calc, cdict)
                            options["calculations"].append(cdict)
                    elif mode == "pscale":
                        for temp in temperature:
                            cdict = {}
                            cdict["mode"] = calc["mode"]
                            if not len(pressure)==2:
                                raise ValueError("At least two pressure values are needed for pscale")
                            cdict["pressure"] = pressure[0]
                            cdict["pressure_stop"] = pressure[-1]
                            cdict["temperature"] = temp
                            cdict["temperature_stop"] = temp
                            cdict["lattice"] = lat
                            if state[i] in ['solid', 'liquid']:
                                cdict["state"] = state[i]
                            else:
                                raise ValueError('state has to be either solid or liquid')
                            cdict["nelements"] = options["nelements"]
                            cdict["element"] = options["element"]
                            cdict["lattice_constant"] = lattice_constant[i]
                            cdict["iso"] = iso[i]
                            cdict["fix_lattice"] = fix_lattice[i]
                            cdict = prepare_optional_keys(calc, cdict)
                            options["calculations"].append(cdict)                                            
                    else:
                        for press in pressure:
                            for temp in temperature:
                                cdict = {}
                                cdict["mode"] = calc["mode"]
                                cdict["temperature"] = temp
                                cdict["pressure"] = press
                                cdict["lattice"] = lat
                                if state[i] in ['solid', 'liquid']:
                                    cdict["state"] = state[i]
                                else:
                                    raise ValueError('state has to be either solid or liquid')

                                cdict["temperature_stop"] = temp
                                cdict["nelements"] = options["nelements"]
                                cdict["element"] = options["element"]
                                cdict["lattice_constant"] = lattice_constant[i]
                                cdict["baro"] = baro[i]
                                cdict["fix_lattice"] = fix_lattice[i]
                                cdict = prepare_optional_keys(calc, cdict)
                                options["calculations"].append(cdict)

                                if mode == "alchemy":
                                    #if alchemy mode is selected: make sure that hybrid pair styles
                                    if not len(options["md"]["pair_style"]) == 2:
                                        raise ValueError("Two pair styles need to be provided")
    return options

def create_identifier(calc):
    """
    Generate an identifier

    Parameters
    ----------
    calc: dict
        a calculation dict

    Returns
    -------
    identistring: string
        unique identification string
    """
    #lattice processed
    prefix = calc["mode"]

    if prefix == 'melting_temperature':
        ts = int(0)
        ps = int(0)

        l = 'tm'

    else:
        ts = int(calc["temperature"])
        ps = int(calc["pressure"])

        l = calc["lattice"]
        l = l.split('/')
        l = l[-1]


    identistring = "-".join([prefix, l, str(ts), str(ps)])
    return identistring

def read_inputfile(file):
    """
    Read calphy inputfile

    Parameters
    ----------
    file : string
        input file

    Returns
    -------
    options : dict
        dictionary containing input options

    """
    options = read_yamlfile(file)

    for i in range(len(options["calculations"])):
        identistring = create_identifier(options["calculations"][i])
        options["calculations"][i]["directory"] = identistring

    return options