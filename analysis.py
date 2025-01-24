import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')  # Importing the given dataset

# Convert SMILES to ECFP
def smiles_to_ecfp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    else:
        return None

df['ECFP'] = df['SMILES'].apply(smiles_to_ecfp)

# Prepare data for training
X = pd.DataFrame(df['ECFP'].tolist())
y = df['avg_lifespan_change_percent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Save promising compounds
df['Predicted Lifespan Impact'] = model.predict(pd.DataFrame(df['ECFP'].tolist()))
promising = df[df['Predicted Lifespan Impact'] >= 10]
promising.to_csv('promising_compounds.csv', index=False)

# Plot results
sns.barplot(x='compound_name', y='Predicted Lifespan Impact', data=promising)
plt.axhline(10, color='r', linestyle='--', label='Threshold: 10%')
plt.legend()
plt.title("Promising Compounds")
plt.xticks(rotation=45)
plt.savefig('output_plots/promising_compounds.png')  # Save to the correct folder
plt.show()
