from logging import config
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from google.generativeai import types
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
from google import genai
from google.genai import types
#from googlesearch import search
import time
import pathlib
import pandas as pd
import psi4
import psi4
from psi4.driver.procrouting.response.scf_response import tdscf_excitations
import requests

smiles = 'CN(C)C1=C(C(CCC1)(C)C)/C=C/C(=C/C=C\C(=C/C=O)\C)/C'

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


psi4_geometry_string = f"""
{stripped_xyz}
"""

mol_psi4 = psi4.geometry(psi4_geometry_string)

psi4.core.set_output_file("molecule_out")
psi4.core.set_num_threads(4)

# Set method and basis set
psi4.set_options({
    "save_jk": True,
    "basis": "6-31g*",
    "tdscf_states": 5  # Number of excited states to compute
})

# Run TD-DFT (Time-Dependent DFT)
e, wfn = psi4.energy("HF/cc-pVDZ", return_wfn=True, molecule=mol_psi4)
res = tdscf_excitations(wfn, states=10)


for k, v in res[0].items():
    print(f"{k} = {v}")
    
first_key, first_value = next(iter(res[0].items()))
print(f"{first_key} = {first_value}")

print(first_value)

print("Finished TDHF excitation calculation.")