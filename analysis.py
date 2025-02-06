import pandas as pd
import numpy as np
import pubchempy as pcp
import ssl

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

# Disable SSL verification (only if you have certificate issues)
ssl._create_default_https_context = ssl._create_unverified_context

# 1) Define helper functions
def get_smiles_from_pubchem(compound_name):
    """Fetch canonical SMILES for a compound name from PubChem."""
    try:
        compound = pcp.get_compounds(compound_name, 'name')
        if compound:
            return compound[0].canonical_smiles
    except Exception as e:
        print(f"Error fetching SMILES for {compound_name}: {e}")
    return None

def smiles_to_ecfp(smiles, radius=2, n_bits=2048):
    """Generate ECFP (Morgan) fingerprint as a list of bits (0/1) from a SMILES string."""
    try:
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
                fp = generator.GetFingerprint(mol)
                return list(fp)
            else:
                print(f"Invalid SMILES: {smiles}")
        else:
            print("SMILES is None, skipping...")
    except Exception as e:
        print(f"Error generating ECFP for SMILES {smiles}: {e}")
    return None

# 2) Load the full dataset
df = pd.read_csv('dataset.csv')

# 3) Take a small subset (first 5 rows) just for a test run
df_small = df.iloc[:2000].copy()

# Create columns for SMILES and ECFP
df_small['smiles'] = None
df_small['ecfp'] = None

index = 0 
# 4) Populate SMILES and ECFP in this small subset
for idx, row in df_small.iterrows():
    compound_name = row['compound_name']
    smiles = get_smiles_from_pubchem(compound_name)
    df_small.at[idx, 'smiles'] = smiles
    print(index)
    index += 1 
    if smiles:
        ecfp_bits = smiles_to_ecfp(smiles)
        df_small.at[idx, 'ecfp'] = ecfp_bits

# 5) Define our target column
target_col = 'avg_lifespan_change_percent'

# 6) Drop any rows missing the target or missing ECFP
df_small = df_small.dropna(subset=[target_col])
df_small = df_small[df_small['ecfp'].notnull()]

# If we still have at least 2-3 rows, we can attempt a tiny model
if len(df_small) < 2:
    print("Not enough rows to train/test a model. Here's what's left:")
    print(df_small)
else:
    # 7) Build feature matrix (X) and target vector (y)
    X_list = []
    y_list = []
    for _, row in df_small.iterrows():
        X_list.append(row['ecfp'])  # 2048-bit ECFP
        y_list.append(row[target_col])

    X = np.array(X_list)
    y = np.array(y_list)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X sample (first row):", X[0][:20], "...")  # Show first 20 bits as a preview
    print("y sample (first row):", y[0])

    # 8) Build and evaluate a Random Forest with cross-validation
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    kfold = KFold(n_splits=2, shuffle=True, random_state=42)  # with just a few points, 2-fold is about all we can do

    scores_mae = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
    mae_scores = -scores_mae  # turn negative into positive
    print("MAE (CV folds):", mae_scores)
    print("Mean MAE:", mae_scores.mean())
