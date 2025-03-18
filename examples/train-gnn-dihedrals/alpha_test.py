# %%
import os
#path = os.getcwd()
path = '/home/cree/code/torsion-nagl/off-openff-nagl/examples/train-gnn-dihedrals'
os.chdir(path) 

# %%
import collections
import tqdm

from rdkit import Chem

from qcportal import PortalClient
from openff.units import unit

from openff.toolkit import Molecule, ForceField
from openff.qcsubmit.results import BasicResultCollection
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.esp.qcresults import from_qcportal_results
from openff.recharge.grids import MSKGridSettings
from openff.recharge.utilities.geometry import compute_vector_field

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

import torch
import MDAnalysis as mda

rdmol = Chem.SDMolSupplier(f'{path}/eclipse_ClCCCl.sdf', removeHs=False)[0]
test_molecule = Molecule.from_rdkit(rdmol)

rdmol

# %%
from openff.nagl.nn.postprocess import PostprocessLayer
from openff.nagl.nn._pooling import PoolProperTorsionFeatures, PoolBondFeatures
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import rdMolTransforms
import torch


# directly extracted from nagl

#n_features: int = 1
# embeddings of eclipse_ClCCCl.sdf calculated by nagl
c_ij = torch.tensor([-0.1059, -0.1241, -0.1241, -0.1059, -0.1080, -0.1080, -0.1080, -0.1080,
    -0.1241, -0.1241, -0.1241, -0.1241, -0.1241, -0.1241, -0.1241, -0.1241,
    -0.1241, -0.1241])

# d_ij = self._pooling_layer._calculate_internal_coordinates(molecule)
d_ij = torch.tensor([-2.7376, -2.7376,  0.4040,  0.4035, -2.7381, -2.7381, -2.7376,  0.4040,
    -2.7381, -2.7381,  0.4040, -2.7376, -2.7376, -2.7376, -2.7381,  0.4035,
        0.4035, -2.7381])

c_ij_np = c_ij.numpy()
d_ij_np = d_ij.numpy()

def forward(
    molecule,
    #inputs: torch.Tensor,
    **kwargs
):
    '''given an embedding, calculate the torsion angles, nagl version'''

    s_ij = torch.empty((2, *d_ij.shape), dtype=d_ij.dtype)
    s_ij[0, :] = torch.cos(d_ij)
    s_ij[1, :] = torch.sin(d_ij)
    s_ij = s_ij.T # (n_torsions, 2)

    # filter by central bond
    proper_torsion_indices_T = torch.tensor([[ 0,  0,  0,  1,  1,  1,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8],
        [ 1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  1,  1,  1,  1,  2,  2,  2,  2],
        [ 2,  2,  2,  3,  3,  3,  1,  1,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3],
        [ 3,  7,  8,  4,  9, 10,  5,  6,  7,  8,  7,  8,  7,  8,  9, 10,  9, 10]])
    #print(proper_torsion_indices_T)
    bond_indices = defaultdict(list)
    for i, atom_2 in enumerate(
        proper_torsion_indices_T[1]
    ):
        atom_3 = proper_torsion_indices_T[2][i]
        bond = tuple(sorted([atom_2.item(), atom_3.item()]))
        bond_indices[bond].append(i)

    a_dict = {}
    
    for bond, indices in bond_indices.items():
        s = torch.sum(
            c_ij[indices].reshape((-1, 1)) * s_ij[indices],
            dim=0
        )
        #print(s)
        s_norm = torch.norm(s)
        #print(f's: {s}, s_norm: {s_norm}')
        a = torch.arctan2(*(s / s_norm))
        print(f'bond {bond}, nagl alpha: {a}')
        a_dict[bond] = a
    
    # ... set bonds that aren't central bonds in torsions to 0?

    # Get all bonds from molecule
    all_bond_indices = [(bond.atom1_index, bond.atom2_index) for bond in molecule.bonds]
    # Sort each bond tuple to match the format used in a_dict
    all_bond_indices = [tuple(sorted(bond)) for bond in all_bond_indices]
    
    # Set bonds that aren't central bonds in torsions to 0
    for bond in all_bond_indices:
        if bond not in a_dict:
            a_dict[bond] = torch.tensor(0.0)

    # sort
    sorted_keys = sorted(a_dict)
    alphas = torch.empty((len(a_dict),)).flatten()
    for i, key in enumerate(sorted_keys):
        alphas[i] = a_dict[key].item()
    
    return alphas

# inputs = torch.tensor([[-0.1059],
#         [-0.1241],
#         [-0.1241],
#         [-0.1059],
#         [-0.1080],
#         [-0.1080],
#         [-0.1241],
#         [-0.1241],
#         [-0.1241],
#         [-0.1241],
#         [-0.1241],
#         [-0.1241],
#         [-0.1241]])
#a: -1.9849530458450317

alpha = forward(test_molecule)

# %%


# %%

import numpy as np

pi = np.pi

# # loop through all torsions for C -- C bond
# delta_ij = np.array([0, 2*pi/3, 4*pi/3, 4*pi/3, 0, 2*pi/3, 2*pi/3, 4*pi/3, 0])

# #c_hidx,hidx
# c_25 = 4
# c_26 = 2
# c_27 = 2
# c_35 = 2
# c_36 = 1
# c_37 = 1
# c_45 = 2
# c_46 = 1
# c_47 = 1

# #c_25 = 2
# #c_26 = 4
# #c_27 = 2
# #c_35 = 1
# #c_36 = 2
# #c_37 = 1
# #c_45 = 1
# #c_46 = 2
# #c_47 = 1

# c_ij = np.array([c_25, c_26, c_27, c_35, c_36, c_37, c_45, c_46, c_47])

def calc_s(delta_ij, c_ij):
    '''a simple test calculation, given ∆_(ij) returns a vector [cos(∆_(ij)), sin(∆_(ij))], s'''
    # turn into a 9X2 array
    s_ij = np.vstack((np.cos(delta_ij), np.sin(delta_ij))).T 
    
    cs_ij = c_ij[:, np.newaxis] * s_ij
    #print(cs_ij)
    #print(f'c*s = {s_ij}')
    s = np.sum(cs_ij, axis=0)
    return s

def calc_alpha(s):
    '''a simple test calculation, given s returns a torsion angle, alpha'''
    r = np.sqrt(s[0]**2 + s[1]**2)
    #alpha = np.arctan2(r*np.cos(s[0]/r), r*np.sin(s[1]/r)) # probably wrong
    alpha = np.arctan2(s[0] / r, s[1] / r)
    
    return alpha

# rotate by some amount
gamma = 0.1

#rotated_delta_ij = delta_ij + gamma
#print(f'original torsions: {delta_ij}, \nrotated torsions: {rotated_delta_ij}')


def test_embedding(c_ij, delta_ij):
    '''a simple test calculation of alpha given an embedding'''
    s = calc_s(delta_ij, c_ij)
    #print(s)
    #s_rotated = calc_s(rotated_delta_ij, c_ij)
    #print(f'original s: {s}, \nrotated s: {s_rotated}')

    a = calc_alpha(s)
    return a

test_a = test_embedding(c_ij_np, d_ij_np)

print('alpha geometry test function:', test_a)


# %%
