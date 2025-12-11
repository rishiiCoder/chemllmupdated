import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

# 1️⃣ Define SMILES string
smiles = "O" 
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)  # add hydrogens
AllChem.EmbedMolecule(mol, AllChem.ETKDG())
AllChem.UFFOptimizeMolecule(mol)

# 2️⃣ Generate XYZ coordinates
conf = mol.GetConformer()
xyz_lines = []
for atom in mol.GetAtoms():
    pos = conf.GetAtomPosition(atom.GetIdx())
    xyz_lines.append(f"{atom.GetSymbol():2} {pos.x: .6f} {pos.y: .6f} {pos.z: .6f}")
xyz_string = "\n".join(xyz_lines)

# 3️⃣ Create ORCA input file content
orca_input = f"""! BP86 def2-SVP TightSCF

%tddft
  Nroots   10
  triplets true
end

* xyz 0 1
{xyz_string}
*
"""

input_file = "molecule.inp"
output_file = "molecule.out"

# 4️⃣ Write ORCA input file
with open(input_file, "w") as f:
    f.write(orca_input)

# 5️⃣ Run ORCA
print("Running ORCA...")
subprocess.run(["orca", input_file], stdout=open(output_file, "w"))
print("ORCA finished.")

# 6️⃣ Parse output for absorption/CD sections
with open(output_file, "r") as f:
    data = f.read()

sections = [
    ("Electric dipole absorption", "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS",
                                    "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS"),
    ("Velocity dipole absorption", "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS",
                                    "CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"),
    ("CD spectrum electric dipole", "CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS",
                                    "CD SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS")
]

for name, start_key, end_key in sections:
    start = data.find(start_key)
    end = data.find(end_key) if end_key else None
    print(f"\n=== {name} ===")
    if start != -1:
        print(data[start:end])
    else:
        print("Section not found.")
