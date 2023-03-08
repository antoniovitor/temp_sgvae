from subprocess import Popen
from random import randint
from pathlib import Path
import sys, os


xyz = ['CCCOCCSN[C-](F)CC','CCC[C-](CCN)C(C)C','CCC[BH-](N)C=O','CCOCOCC[AlH-](F)C(F)CC(F)CF','C=C(C[InH-](CCCC)C(=O)OOC)ONC','CCC(NNSC)[BH-](CCN)CCN','N=C(N)NCCCC[N-][SH](=O)=O','CCCC[AsH3-](CCCOCC)CC(=O)O','CSCC[BH-](S(=O)OOC(F)F)S(=O)(=O)O','C[PH-](C)C']
for x in xyz:
	aux = x + '.xyz'
	
	cmd = f'jmol {aux}'.replace('(', '\(').replace(')', '\)')
	# print(os.path.basename(xyz[idx]).split('.')[0])
	Popen(cmd, shell=True)