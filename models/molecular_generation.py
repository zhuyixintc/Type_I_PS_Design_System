import random
from rdkit import Chem
from models.utils import s2m, m2s, atom_number_count, remove_dp
from rdkit.Chem.Draw import rdDepictor
from rdkit import RDLogger


rdDepictor.SetPreferCoordGen(True)
RDLogger.DisableLog('rdApp.*')


# get all possible valid next state
def get_valid_trajectory(state):
    action_space = []

    # load all available fragments
    f = open()  # your fragment library
    frag_pool = f.read()
    frag_pool = frag_pool.split("\n")
    frag_pool.pop(0)
    frag_pool.pop()
    f.close()

    # add one more fragment
    if '*' in state:
        for frag_smi in frag_pool:
            x1 = s2m(state)
            x2 = s2m(frag_smi)
            combo = Chem.CombineMols(x1, x2)

            # get main atoms
            atom1 = []
            for atom in x1.GetAtoms():
                atom1.append(
                    [atom.GetIdx(), atom.GetSymbol(),
                     [(nbr.GetIdx(), nbr.GetSymbol()) for nbr in atom.GetNeighbors()]])
            atom1_symbol = [elt[1] for elt in atom1]
            idx1 = [o for o, p in list(enumerate(atom1_symbol)) if p == '*']

            # get combo atoms
            atom_combo = []
            for atom in combo.GetAtoms():
                atom_combo.append(
                    [atom.GetIdx(), atom.GetSymbol(),
                     [(nbr.GetIdx(), nbr.GetSymbol()) for nbr in atom.GetNeighbors()]])
            atom_combo_symbol = [elt[1] for elt in atom_combo]
            idx_combo = [o for o, p in list(enumerate(atom_combo_symbol)) if p == '*']
            idx2 = idx_combo[len(idx1):]

            # generate molecules
            for i in range(len(idx1)):
                for j in range(len(idx2)):
                    edcombo = Chem.EditableMol(combo)

                    # get linking index and add single bond
                    x = idx1[i]
                    y = idx2[j]
                    p1 = atom_combo[x][2][0][0]
                    p2 = atom_combo[y][2][0][0]
                    edcombo.AddBond(p1, p2, order=Chem.rdchem.BondType.SINGLE)

                    # remove predefined linking points (should start from the highest order)
                    atom_to_remove = [x, y]
                    atom_to_remove.sort(reverse=True)
                    for atom in atom_to_remove:
                        edcombo.RemoveAtom(atom)

                    # back to mol
                    back = edcombo.GetMol()
                    back_smi = m2s(back)

                    if '*' in back_smi:
                        if atom_number_count(remove_dp(back_smi)) <= 70:
                            action_space.append(back_smi)
                    else:
                        if atom_number_count(back_smi) <= 70:
                            action_space.append(back_smi)

    # stay no modification
    action_space.append(state)

    action_space = list(dict.fromkeys(action_space))

    return action_space


def initialise_state():
    f = open()  # your fragment library
    frag_pool = f.read()
    frag_pool = frag_pool.split("\n")
    frag_pool.pop(0)
    frag_pool.pop()
    f.close()
    starting_smiles = random.sample(frag_pool, 1)[0]
    return starting_smiles

