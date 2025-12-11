import psi4
from psi4.driver.procrouting.response.scf_response import tdscf_excitations

psi4.core.set_output_file("molecule_out")
psi4.core.set_num_threads(4)

# Define the molecule
mol = psi4.geometry("""
0 1
C  0.008747  1.292422  0.051097
C  1.310292  1.825512  0.204111
C  2.456455  1.029926  0.183693
C  3.552191  1.843729  0.346968
C  3.445569  3.272902  0.493089
C  2.121920  3.938860  0.438330
C  1.022805  3.128418  0.306810
C  -0.088269  4.025658  0.293566
C  -1.380372  3.176351  0.112691
C  -1.268282  1.718895  0.008484
C  -0.164418  0.954819  -0.017336
O  4.584013  0.875126  0.454942
H  -0.142030  1.617543  1.041859
H  0.124792  0.437612  -0.712520
H  0.926826  1.652445  1.115086
H  1.910354  1.740669  -0.738535
H  4.235939  1.683854  1.079187
H  2.048217  4.999719  0.500535
H  0.322341  3.045538  -0.664466
H  -1.948124  4.034355  0.123004
H  -2.102319  2.665853  -0.567859
H  -1.931040  1.365664  0.929953
H  -1.729437  1.164432  -0.756473
H  -0.123652  5.080462  0.430283
""")

# Set method and basis set
psi4.set_options({
    "save_jk": True,
    "basis": "6-31g*",
    "tdscf_states": 5  # Number of excited states to compute
})

# Run TD-DFT (Time-Dependent DFT)
e, wfn = psi4.energy("HF/cc-pVDZ", return_wfn=True, molecule=mol)
res = tdscf_excitations(wfn, states=10)


for k, v in res[0].items():
    print(f"{k} = {v}")
    
first_key, first_value = next(iter(res[0].items()))
print(f"{first_key} = {first_value}")

print(first_value)