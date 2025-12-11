from rdkit import Chem
from rdkit.Chem import Draw

# Define SMILES strings
retinal_smiles = r"CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C\\C(=C\\C=O)\\C)/C"
modified_smiles_one =  r'CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C\\C(=C\\C=C\\C=C-C#N)\\C)/C'
modified_smiles_two = r'CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C\\C(=C\\C=C-C#N)\\C)/C'
modified_smiles_three = r'CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C\\C(=C\\C1=CNC=C1)\\C)/C'

# Convert to RDKit molecule objects
retinal_mol = Chem.MolFromSmiles(retinal_smiles)
modified_smiles_one = Chem.MolFromSmiles(modified_smiles_one)
modified_smiles_two = Chem.MolFromSmiles(modified_smiles_two)
modified_smiles_three = Chem.MolFromSmiles(modified_smiles_three)

# Draw both molecules side by side
Draw.MolsToImage([retinal_mol, modified_smiles_one, modified_smiles_two, modified_smiles_three], legends=["Retinal", "modified_smiles_one", "modified_smiles_two", "modified_smiles_three"])