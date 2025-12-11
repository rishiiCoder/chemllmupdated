import psi4
from rdkit import Chem
from rdkit.Chem import AllChem

# 1. Define the SMILES string
smiles = 'CCO' # Ethanol example

# 2. Generate a 2D molecule object from the SMILES
mol_rdkit = Chem.MolFromSmiles(smiles)

# 3. Add explicit hydrogens and generate 3D coordinates using the MMFF94 force field
mol_rdkit = Chem.AddHs(mol_rdkit)
AllChem.EmbedMolecule(mol_rdkit, AllChem.ETKDGv3()) # Use a more modern conformer generation method
AllChem.MMFFOptimizeMolecule(mol_rdkit) # Optimize with a force field

# 4. Convert the RDKit molecule to an XYZ string format
mol_xyz = Chem.rdmolfiles.MolToXYZBlock(mol_rdkit)

# 5. Get rid of the first two lines of the XYZ string (which contain atom count and a comment line)
stripped_xyz = mol_xyz.split('\n', 2)[2:][0]

# 6. Load the coordinates into a Psi4 molecule object
# The format for the geometry string needs to be correct for Psi4
psi4_geometry_string = f"""
{stripped_xyz}
"""

mol_psi4 = psi4.geometry(psi4_geometry_string)

# 7. (Optional) Perform a quantum mechanical geometry optimization in Psi4
psi4.set_memory('500 MB')
psi4.set_options({'reference': 'rhf'})
psi4.optimize('scf/cc-pvdz', molecule=mol_psi4)