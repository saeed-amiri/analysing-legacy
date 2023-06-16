"""to save constant constants, paths, nams, which are used in the scripts"""
import typing


# logging files
log: dict[str, typing.Any]
log = {
    'test_log': 'test_log',
}

# files to save or test
files: dict[str, str]
files = {
    'com_pickle': 'com_pickle'
}

# time step (the frequncy of trr file)
times: dict[str, float]
times = {
    'time_step': 100.0
}

# Constant values for reading topol file
topo: dict[str, typing.Any]
topo = {
    'fname': './topol.top',
}

# Number of atoms in each residue make sure you update this for
# different system size
reidues_num: dict[str, int]
reidues_num = {
    'CLA': 1,
    'SOL': 3,
    'ODN': 59,
    'D10': 32,
    'APT_COR': 7924
}

# Giving id to each residues to not save str in array
reidues_id: dict[str, int]
reidues_id = {
    'CLA': 1,
    'SOL': 2,
    'ODN': 3,
    'D10': 4,
    'APT_COR': 5
}

# Nano partcle
np_info: dict[str, typing.Any]
np_info = {
    'radius': 32.6,  # It is in Angstrom
    'np_residues': ['APT', 'COR'],
    'np_name': 'APT_COR',
    'solution_residues': ['CLA', 'SOL', 'ODN', 'D10']
}

# Box information
box: dict[str, typing.Any]
box = {
    'x': 230,
    'y': 230,
    'z': 230,
    'centered': True  # If the center of mass moved to zero
}

# Plot data
plot: dict[str, typing.Any]
plot = {
    'width': 426.79135
}
