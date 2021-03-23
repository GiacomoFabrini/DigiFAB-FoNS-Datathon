import numpy as np
import matplotlib.pyplot as plt 
import scipy as sp
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw 
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
#pd.set_option('max_colwidth', 18)


def fun_group(mol):
    f_groups = {}
    f_groups['aliphatic carboxylic acids'] = Chem.Fragments.fr_Al_COO(mol)
    f_groups['aliphatic hydroxyl groups'] = Chem.Fragments.fr_Al_OH(mol)
    f_groups['aliphatic hydroxyl groups excluding tert-OH'] = Chem.Fragments.fr_Al_OH_noTert(mol)
    f_groups['N functional groups attached to aromatics'] = Chem.Fragments.fr_ArN(mol)
    f_groups['Aromatic carboxylic acide'] = Chem.Fragments.fr_Ar_COO(mol)
    f_groups['aromatic nitrogens'] = Chem.Fragments.fr_Ar_N(mol)
    f_groups['aromatic amines'] = Chem.Fragments.fr_Ar_NH(mol)
    f_groups['aromatic hydroxyl groups'] = Chem.Fragments.fr_Ar_OH(mol)
    f_groups['carboxylic acids'] = Chem.Fragments.fr_COO(mol)
    f_groups['carboxylic acids(COO2)'] = Chem.Fragments.fr_COO2(mol)
    f_groups['carbonyl O'] = Chem.Fragments.fr_C_O(mol)
    f_groups['carbonyl O, excluding COOH'] = Chem.Fragments.fr_C_O_noCOO(mol)
    f_groups['thiocarbonyl'] = Chem.Fragments.fr_C_S(mol)
    f_groups['C(OH)CCN-Ctert-alkyl or C(OH)CCNcyclic'] = Chem.Fragments.fr_HOCCN(mol)
    f_groups['Imines'] = Chem.Fragments.fr_Imine(mol)
    f_groups['Tertiary amines'] = Chem.Fragments.fr_NH0(mol)
    f_groups['Secondary amines'] = Chem.Fragments.fr_NH1(mol)
    f_groups['Primary amines'] = Chem.Fragments.fr_NH2(mol)
    f_groups['hydroxylamine groups'] = Chem.Fragments.fr_N_O(mol)
    f_groups['XCCNR groups'] = Chem.Fragments.fr_Ndealkylation1(mol)
    f_groups['tert-alicyclic amines (no heteroatoms, not quinine-like bridged N)'] = Chem.Fragments.fr_Ndealkylation2(mol)
    f_groups['H-pyrrole nitrogens'] = Chem.Fragments.fr_Nhpyrrole(mol)
    f_groups['thiol groups'] = Chem.Fragments.fr_SH(mol)
    f_groups['aldehydes'] = Chem.Fragments.fr_aldehyde(mol)
    f_groups['alkyl carbamates (subject to hydrolysis)'] = Chem.Fragments.fr_alkyl_carbamate(mol)
    f_groups['alkyl halides'] = Chem.Fragments.fr_alkyl_halide(mol)
    f_groups['allylic oxidation sites excluding steroid dienone'] = Chem.Fragments.fr_allylic_oxid(mol)
    f_groups['amides'] = Chem.Fragments.fr_amide(mol)
    f_groups['amidine groups'] = Chem.Fragments.fr_amidine(mol)
    f_groups['anilines'] = Chem.Fragments.fr_aniline(mol)
    f_groups['aryl methyl sites for hydroxylation'] = Chem.Fragments.fr_aryl_methyl(mol)
    f_groups['azide groups'] = Chem.Fragments.fr_azide(mol)
    f_groups['azo groups'] = Chem.Fragments.fr_azo(mol)
    f_groups['barbiturate groups'] = Chem.Fragments.fr_barbitur(mol)
    f_groups['benzene rings'] = Chem.Fragments.fr_benzene(mol)
    f_groups['benzodiazepines with no additional fused rings'] = Chem.Fragments.fr_benzodiazepine(mol)
    f_groups['Bicyclic groups'] = Chem.Fragments.fr_bicyclic(mol)
    f_groups['diazo groups'] = Chem.Fragments.fr_diazo(mol)
    f_groups['dihydropyridines'] = Chem.Fragments.fr_dihydropyridine(mol)
    f_groups['epoxide rings'] = Chem.Fragments.fr_epoxide(mol)
    f_groups['esters'] = Chem.Fragments.fr_ester(mol)
    f_groups['ether oxygens (including phenoxy)'] = Chem.Fragments.fr_ether(mol)
    f_groups['furan rings'] = Chem.Fragments.fr_furan(mol)
    f_groups['guanidine groups'] = Chem.Fragments.fr_guanido(mol)
    f_groups['halogens'] = Chem.Fragments.fr_halogen(mol)
    f_groups['hydrazine groups'] = Chem.Fragments.fr_hdrzine(mol)
    f_groups['hydrazone groups'] = Chem.Fragments.fr_hdrzone(mol)
    f_groups['imidazole rings'] = Chem.Fragments.fr_imidazole(mol)
    f_groups['imide groups'] = Chem.Fragments.fr_imide(mol)
    f_groups['isocyanates'] = Chem.Fragments.fr_isocyan(mol)
    f_groups['isothiocyanates'] = Chem.Fragments.fr_isothiocyan(mol)
    f_groups['ketones'] = Chem.Fragments.fr_ketone(mol)
    f_groups['ketones excluding diaryl, a,b-unsat. dienones, heteroatom on Calpha'] = Chem.Fragments.fr_ketone_Topliss(mol)
    f_groups['beta lactams'] = Chem.Fragments.fr_lactam(mol)
    f_groups['cyclic esters (lactones)'] = Chem.Fragments.fr_lactone(mol)
    f_groups['methoxy groups -OCH3'] = Chem.Fragments.fr_methoxy(mol)
    f_groups['morpholine rings'] = Chem.Fragments.fr_morpholine(mol)
    f_groups['nitriles'] = Chem.Fragments.fr_nitrile(mol)
    f_groups['nitro groups'] = Chem.Fragments.fr_nitro(mol)
    f_groups['nitro benzene ring substituents'] = Chem.Fragments.fr_nitro_arom(mol)
    f_groups['non-ortho nitro benzene ring substituents'] = Chem.Fragments.fr_nitro_arom_nonortho(mol)
    f_groups['nitroso groups, excluding NO2'] = Chem.Fragments.fr_nitroso(mol)
    f_groups['oxazole rings'] = Chem.Fragments.fr_oxazole(mol)
    f_groups['oxime groups'] = Chem.Fragments.fr_oxime(mol)
    f_groups['para-hydroxylation sites'] = Chem.Fragments.fr_para_hydroxylation(mol)
    f_groups['phenols'] = Chem.Fragments.fr_phenol(mol)
    f_groups['phenolic OH excluding ortho intramolecular Hbond substituents'] = Chem.Fragments.fr_phenol_noOrthoHbond(mol)
    f_groups['phosphoric acid groups'] = Chem.Fragments.fr_phos_acid(mol)
    f_groups['phosphoric ester groups'] = Chem.Fragments.fr_phos_ester(mol)
    f_groups['piperdine rings'] = Chem.Fragments.fr_piperdine(mol)
    f_groups['piperzine rings'] = Chem.Fragments.fr_piperzine(mol)
    f_groups['primary amides'] = Chem.Fragments.fr_priamide(mol)
    f_groups['primary sulfonamides'] = Chem.Fragments.fr_prisulfonamd(mol)
    f_groups['pyridine rings'] = Chem.Fragments.fr_pyridine(mol)
    f_groups['quarternary nitrogens'] = Chem.Fragments.fr_quatN(mol)
    f_groups['thioether'] = Chem.Fragments.fr_sulfide(mol)
    f_groups['sulfonamides'] = Chem.Fragments.fr_sulfonamd(mol)
    f_groups['sulfone groups'] = Chem.Fragments.fr_sulfone(mol)
    f_groups['terminal acetylenes'] = Chem.Fragments.fr_term_acetylene(mol)
    f_groups['tetrazole rings'] = Chem.Fragments.fr_tetrazole(mol)
    f_groups['thiazole rings'] = Chem.Fragments.fr_thiazole(mol)
    f_groups['thiocyanates'] = Chem.Fragments.fr_thiocyan(mol)
    f_groups['thiophene rings'] = Chem.Fragments.fr_thiophene(mol)
    f_groups['unbranched alkanes of at least 4 members (excludes halogenated alkanes)'] = Chem.Fragments.fr_unbrch_alkane(mol)
    f_groups['urea groups'] = Chem.Fragments.fr_urea(mol)
    
    return f_groups

rList = ['C1=NC2=C(N1)C(=O)NC(=O)N2','CN1C2=C(C(=O)NC1=O)NC=N2','CN1C(=O)N(C)c2nc[nH]c2C1=O','CN1C=NC2=C1C(=O)NC(=O)N2C','CN1C=NC2=C1C(=O)N(C)C(=O)N2C',
         'CN1C(=O)N(C)c2nc(Br)n(C)c2C1=O', 'CC(=O)CCCCN1C(=O)C2=C(N=CN2C)N(C1=O)C', 'CCCn1cnc2N(C)C(=O)N(CCCCC(=O)C)C(=O)c12',
         'CC(=O)NC1=CC=C(O)C=C1','NC1=NC(=O)N(C=C1)[C@H]1CC[C@@H](CO)O1',
         'CCC1(CCC(=O)NC1=O)C1=CC=C(N)C=C1','CN1COCN(C1=N[N+](=O)[O-])CC2=CN=C(S2)Cl','OC[C@@H](NC(=O)C(Cl)Cl)[C@H](O)C1=CC=C(C=C1)[N+]([O-])=O',
         'CN1C(C(=O)NC2=NC=CC=C2)=C(O)C2=C(C=CC=C2)S1(=O)=O','COC1=CC2=C(C(OC)=C1OC)C1=CC=C(OC)C(=O)C=C1C(CC2)NC(C)=O',
         '[H][C@@]12CC3=C(C(O)=CC=C3N(C)C)C(=O)C1=C(O)[C@]1(O)C(=O)C(C(N)=O)=C(O)[C@@H](N(C)C)[C@]1([H])C2',
         ]

names = ['Xanthine','3-methlyxanthine','Theophylline','Theobromine','Caffeine',
         '8-bromocaffeine','Pentoxifylline','Propentofylline',
         'Paracetomol','Zalcitabine',
         'Aminoglutethimide','Thiamethoxam','Chloramphenicol',
         'Piroxicam','Colchicine',
         'Minocycline' ]

rMols = [Chem.MolFromSmiles(r) for r in rList]

fg_matrix = []
for i in range(0,len(rMols)):
    fg_matrix.append(fun_group(rMols[i]))
    
fg_matrix = pd.DataFrame(fg_matrix,index = names) 
print(fg_matrix)                    