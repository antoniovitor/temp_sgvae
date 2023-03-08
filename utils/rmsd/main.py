import pandas as pd


from utils.rmsd.utils import *
from matplotlib import pyplot as plt


# ================================
# Defining plot style
# ================================
fontsize = 18
fig_width = 1
plt.rcParams.update(
    get_rcParams(fontsize, fig_width, figsize=(4,4))
)


# ================================
# Loading data
# ================================
cations = pd.read_json(
    '../../DATA/cations_complete.json',
    orient='index',
    precise_float=True
)
anions = pd.read_json(
    '../../DATA/anions_complete.json',
    orient='index',
    precise_float=True
)

# getting molecules with mag equals to 0
cations = cations[cations['mag-fopt'] == 0]
anions = anions[anions['mag-fopt'] == 0]

anions = drop_problematic_mols(anions, '../../DATA/problematic_ions_smiles/all_anions.txt')
cations = drop_problematic_mols(cations, '../../DATA/problematic_ions_smiles/all_cations.txt')


dist = get_rmsd(cations, 'XYZ-MMFF94', 'XYZ-popt')

plt.clf()
plt.hist(dist, bins='auto', color=CATIONS_COLOR)
plt.ylabel('Absolute frequency')
plt.xlabel('RMSD (\AA)')
plt.xticks([0, 1.9, 3.8, 5.7], ['0', '1.9', '3.8', '5.7'])
plt.yticks([0, 250, 500, 750])
plt.ylim([0, 750])
plt.xlim([0, 5.7])
plt.tick_params(direction='in', length=4, right=True, top=True, width=fig_width)
plt.tight_layout()
plt.savefig('cations_RMSD_MMFF94_popt.pdf', bbox_inches='tight', dpi=300)

dist = get_rmsd(cations, 'XYZ-popt', 'XYZ-fopt')

plt.clf()
plt.hist(dist, bins='auto', color=CATIONS_COLOR)
plt.ylabel('Absolute frequency')
plt.xlabel('RMSD (\AA)')
plt.xticks([0, 1.9, 3.8, 5.7], ['0', '1.9', '3.8', '5.7'])
plt.yticks([0, 250, 500, 750])
plt.ylim([0, 750])
plt.xlim([0, 5.7])
plt.tick_params(direction='in', length=4, right=True, top=True, width=fig_width)
plt.tight_layout()
plt.savefig('cations_RMSD_popt_fopt.pdf', bbox_inches='tight', dpi=300)

dist = get_rmsd(cations, 'XYZ-MMFF94', 'XYZ-fopt')

plt.clf()
plt.hist(dist, bins='auto', color=CATIONS_COLOR)
plt.ylabel('Absolute frequency')
plt.xlabel('RMSD (\AA)')
plt.xticks([0, 1.9, 3.8, 5.7], ['0', '1.9', '3.8', '5.7'])
plt.yticks([0, 250, 500, 750])
plt.ylim([0, 750])
plt.xlim([0, 5.7])
plt.tick_params(direction='in', length=4, right=True, top=True, width=fig_width)
plt.tight_layout()
plt.savefig('cations_RMSD_MMFF94_fopt.pdf', bbox_inches='tight', dpi=300)


dist = get_rmsd(anions, 'XYZ-MMFF94', 'XYZ-popt')
plt.clf()
plt.hist(dist, bins='auto', color=ANIONS_COLOR)
plt.ylabel('Absolute frequency')
plt.xlabel('RMSD (\AA)')
plt.xticks([0, 1, 2, 3])
plt.yticks([0, 200, 400, 600])
plt.ylim([0, 600])
plt.xlim([0, 3])
plt.tick_params(direction='in', length=4, right=True, top=True, width=fig_width)
plt.tight_layout()
plt.savefig('anions_RMSD_MMFF94_popt.pdf', bbox_inches='tight', dpi=300)

dist = get_rmsd(anions, 'XYZ-popt', 'XYZ-fopt')

plt.clf()
plt.hist(dist, bins='auto', color=ANIONS_COLOR)
plt.ylabel('Absolute frequency')
plt.xlabel('RMSD (\AA)')
plt.xticks([0, 1, 2, 3])
plt.yticks([0, 200, 400, 600])
plt.ylim([0, 600])
plt.xlim([0, 3])
plt.tick_params(direction='in', length=4, right=True, top=True, width=fig_width)
plt.tight_layout()
plt.savefig('anions_RMSD_popt_fopt.pdf', bbox_inches='tight', dpi=300)

dist = get_rmsd(anions, 'XYZ-MMFF94', 'XYZ-fopt')

plt.clf()
plt.hist(dist, bins='auto', color=ANIONS_COLOR)
plt.ylabel('Absolute frequency')
plt.xlabel('RMSD (\AA)')
plt.xticks([0, 1, 2, 3])
plt.yticks([0, 200, 400, 600])
plt.ylim([0, 600])
plt.xlim([0, 3])
plt.tick_params(direction='in', length=4, right=True, top=True, width=fig_width)
plt.tight_layout()
plt.savefig('anions_RMSD_MMFF94_fopt.pdf', bbox_inches='tight', dpi=300)
