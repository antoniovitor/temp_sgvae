from openbabel import pybel
from utils.xyz import xyz2pybel

def compare_geometries(xyz_path_1, xyz_path_2):
    xyz1 = xyz2pybel(xyz_path_1)
    xyz2 = xyz2pybel(xyz_path_2)

    includeH=True
    symmetry=False
    aligner = pybel.ob.OBAlign(xyz1, xyz2, includeH, symmetry)
    aligner.Align()

    return aligner.GetRMSD()