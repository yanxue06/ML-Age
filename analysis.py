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

def extract_numeric_value(dosage_str):
    """Extract numeric value from dosage string"""
    if pd.isna(dosage_str):
        return None
    match = re.search(r'([\d.]+)', str(dosage_str))
    return float(match.group(1)) if match else None

def process_additional_features(df):
    """Process additional features for the model"""
    
    # Create new features DataFrame
    additional_features = pd.DataFrame()
    
    # 1. Process dosage - extract numeric values
    additional_features['dosage_value'] = df['dosage'].apply(extract_numeric_value)
    
    # 2. Convert significance to numeric
    le = LabelEncoder()
    additional_features['significance_encoded'] = le.fit_transform(df['avg_lifespan_significance'].fillna('NA'))
    
    # 3. Encode gender
    additional_features['is_male'] = df['gender_new'].apply(lambda x: 1 if str(x).lower() == 'male' else 0)
    additional_features['is_female'] = df['gender_new'].apply(lambda x: 1 if str(x).lower() == 'female' else 0)
    
    # 4. Encode species
    species_dummies = pd.get_dummies(df['species'], prefix='species')
    additional_features = pd.concat([additional_features, species_dummies], axis=1)
    
    # 5. Fill missing values with median
    additional_features = additional_features.fillna(additional_features.median())
    
    return additional_features

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

# Process additional features
additional_features_df = process_additional_features(df_small)

# Combine ECFP with additional features
X_list = []
y_list = []

for idx, row in df_small.iterrows():
    if row['ecfp'] is not None:
        # Get ECFP features
        ecfp_features = np.array(row['ecfp'])
        
        # Get additional features
        additional_feat = additional_features_df.iloc[idx].values
        
        # Combine features
        combined_features = np.concatenate([ecfp_features, additional_feat])
        
        X_list.append(combined_features)
        y_list.append(row[target_col])

X = np.array(X_list)
y = np.array(y_list)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create a more robust model with more trees and better parameters
model = RandomForestRegressor(
    n_estimators=100,  # More trees
    max_depth=10,      # Control depth to prevent overfitting
    min_samples_split=5,
    random_state=42
)

# Use 3-fold CV instead of 2-fold for better evaluation
kfold = KFold(n_splits=3, shuffle=True, random_state=42)

# 8) Build and evaluate a Random Forest with cross-validation
scores_mae = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
mae_scores = -scores_mae  # turn negative into positive
print("MAE (CV folds):", mae_scores)
print("Mean MAE:", mae_scores.mean())



