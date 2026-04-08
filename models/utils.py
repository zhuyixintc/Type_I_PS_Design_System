from rdkit.Chem import AllChem, Descriptors
from rdkit import Chem
import numpy as np


# get mol from smiles
def s2m(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol


# get smiles from mol
def m2s(mol):
    smiles = Chem.MolToSmiles(mol)
    return smiles


# atom number count
def atom_number_count(smi):
    mol = s2m(smi)
    atom_number = 0
    for _ in mol.GetAtoms():
        atom_number += 1
    return atom_number


def remove_dp(smi):
    mol = s2m(smi)
    prod = Chem.RemoveHs(AllChem.ReplaceSubstructs(mol, s2m('*'), s2m('[H]'), True)[0])
    prod_smi = m2s(prod)
    return m2s(s2m(prod_smi))


# get morgan fingerprint as input feature
def get_input_feature(smiles):
    m = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048)
    fp = list(fp)
    return fp

