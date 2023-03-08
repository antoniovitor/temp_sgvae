import numpy as np


from openbabel import pybel
from tqdm import tqdm


# ================================
# Defining the color palette
# ================================
DEFAULT_COLOR = '#FF5722'
CATIONS_COLOR = '#3F51B5'
ANIONS_COLOR  = '#43A047'


def get_rcParams(fontsize=20, width=1, figsize=(10,10)):
    """
    Default style to Matplotlib's figure.

    Parameters
    ----------
    font_size: int, default 20
        Set the font size
    width: float, default 1
        Set the line width
    figsize: list or tuple, default (10,10)
        Set the figure size
    """

    return {
        'axes.labelsize': fontsize,
        'axes.linewidth': width,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'figure.facecolor': 'white',
        'font.family' : 'CMU Serif',
        'text.usetex': True, 
        'text.latex.preamble': r'\usepackage{siunitx, mhchem}',
        'axes.titlesize': fontsize,
        'legend.fontsize': fontsize,
        'font.family': 'serif',
        'font.size': fontsize,
        'grid.color' : 'gray',
        'figure.figsize' : figsize
    }


def drop_problematic_mols(df, file):
    """
    Removes problematic molecules from a dataframe

    Parameters
    ----------
    df: dataframe-like object
        Dataframe with the molecule's data
    file: str
        File with DataFrame's indices to drop
    """
    
    with open(file, 'r') as f:
        problematic = [i.strip() for i in f.readlines()]
    return df.drop(problematic)
    
    
def get_rmsd(df, ref, target):
    """
    Compute the root-mean-square deviation (RMSD) between conformations

    Parameters
    ----------
    df: dataframe-like object
        Dataframe with the molecule's data
    ref: str
        Reference conformation
    target: str
        Target conformation
    """

    def xyz2pybel(xyz):

        xyz_str = '{}\n\n'.format(len(xyz))
        for atom in xyz:
            xyz_str += '{}\t{}\t{}\t{}\n'.format(atom[0],atom[1],atom[2],atom[3])

        obconversion = pybel.ob.OBConversion()
        _ = obconversion.SetInFormat('xyz')
        mol = pybel.ob.OBMol()
        _ = obconversion.ReadString(mol, xyz_str)

        return mol

    diff = []
    for _, (ini, out) in df[[ref, target]].iterrows():

        aligner = pybel.ob.OBAlign(xyz2pybel(ini), xyz2pybel(out), True, False)
        _ = aligner.Align()
        diff.append(aligner.GetRMSD())
        
    return diff
