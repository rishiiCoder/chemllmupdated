import psi4


print("running")
# Define molecule (water geometry in Angstroms)
h2o = psi4.geometry("""
0 1
O  0.000000  0.000000  0.000000
H  0.000000  0.757160  0.586260
H  0.000000 -0.757160  0.586260
""")

# Set memory and output file
psi4.set_memory('500 MB')
psi4.core.set_output_file('output.dat', False)

# Set calculation options
psi4.set_options({'basis': '6-31G', 'scf_type': 'pk'})

# Run a DFT energy calculation
energy = psi4.energy('B3LYP')

# Print the energy
print(f"Computed energy: {energy} Hartree")

