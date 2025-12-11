import subprocess
import cclib
import re

input_file = "molecule.inp"
output_file = "molecule.out"

# After ORCA finishes, read and print the absorption section
with open(output_file, "r") as f:
    data = f.read()

# Find the start and end of the absorption section
# Electric dipole absorption
start = data.find("ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS")
end   = data.find("ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS")
print(data[start:end])

# Velocity dipole absorption
start = data.find("ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS")
end   = data.find("CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS")
print(data[start:end])

# CD spectrum electric dipole
start = data.find("CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS")
end   = data.find("CD SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS")
print(data[start:end])

# CD spectrum velocity dipole
start = data.find("CD SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS")
# if this is the last section, you can just print to the end of the file
print(data[start:])

if start != -1 and end != -1:
    absorption_section = data[start:end]
    print(absorption_section)
else:
    print("Absorption section not found in the output file.")
    

