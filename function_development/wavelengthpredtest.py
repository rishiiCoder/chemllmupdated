

# result = predict_peak(some_input)
import pandas as pd

# Replace this with your molecule's SMILES string
smiles = "CCO"  

# If you know the true absorption peak, put it here; otherwise, None or leave blank
absorption_peak_nm_expt = None  

# Create a DataFrame with one row
#df = pd.DataFrame({
   # "smiles": [smiles],
   # "absorption_peak_nm_expt": [absorption_peak_nm_expt]
#})

# Save to CSV (no index column)
#df.to_csv("single_molecule.csv", index=False)

#print("CSV file 'single_molecule.csv' created!")
#python uvvisml/predict.py --test_file single_mol.csv --property absorption_peak_nm_expt --method chemprop --preds_file single_mol_pred.csv

test_file = 'uvvisml/uvvisml/data/splits/lambda_max_abs/deep4chem/group_by_smiles/smiles_target_test.csv'
df = pd.read_csv(test_file)
print(df)
#python uvvisml/uvvisml/predict.py --test_file uvvisml/uvvisml/data/splits/lambda_max_abs/deep4chem/group_by_smiles/smiles_target_test.csv --property absorption_peak_nm_expt --method chemprop --preds_file uvvisml/uvvisml/data/splits/lambda_max_abs/deep4chem/group_by_smiles/smiles_target_test.csv