import psi4

psi4.core.set_output_file('output.dat', False)

# Define the molecule
mol = psi4.geometry("""
0 1
C 0.0 0.0 0.0
O 0.0 0.0 1.2
symmetry c1
""")

# Set calculation options
psi4.set_options({
    'basis': '6-31G*',
    'scf_type': 'pk',
    'reference': 'rhf'
})

# Run DFT calculation (B3LYP functional)
energy, wfn = psi4.energy('B3LYP', return_wfn=True)

# Extract HOMO, LUMO, and gap
eps_a = wfn.epsilon_a().to_array()
nocc = wfn.nalpha()
homo = eps_a[nocc - 1]
lumo = eps_a[nocc]
gap = lumo - homo


import psi4

mol = psi4.geometry("""
0 1
C 0.0 0.0 0.0
O 0.0 0.0 1.2
symmetry c1
""")

psi4.set_options({'basis': '6-31G*', 'reference': 'rhf'})

# Neutral molecule
E_neutral = psi4.energy('B3LYP/6-31G*')

# Cation (+1 charge) — remove 1 electron
mol.charge = 1
E_cation = psi4.energy('B3LYP/6-31G*')

# Anion (-1 charge) — add 1 electron
mol.charge = -1
E_anion = psi4.energy('B3LYP/6-31G*')

# Ionization potential (IP) and electron affinity (EA)
IP = (E_cation - E_neutral) * 27.2114   # eV
EA = (E_neutral - E_anion) * 27.2114    # eV

print("DATA ON ELECTRON ACCEPTION AND DONATION")
print(f"HOMO: {homo:.3f} Hartree = {homo * 27.2114:.3f} eV")
print(f"LUMO: {lumo:.3f} Hartree = {lumo * 27.2114:.3f} eV")
print(f"Gap:  {gap:.3f} Hartree = {gap * 27.2114:.3f} eV")
print(f"Ionization Potential (IP): {IP:.3f} eV")
print(f"Electron Affinity (EA): {EA:.3f} eV")

