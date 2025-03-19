# ML-Age 

## Overview
This project explores the lifespan extension potential of chemical compounds using machine learning. Under the mentorship of [Dr. Michał Koziarski](https://koziarskilab.com/), we are developing predictive models to assess compounds' effects on longevity.

## Approach
```markdown
1. Extract compound names from the dataset and retrieve SMILES notations via PubChem.
2. Convert SMILES into numerical representations using RDKit to generate Extended-Connectivity Fingerprints (ECFPs).
3. Train a **Random Forest Regressor** to predict lifespan extension percentages.
4. Evaluate performance using **k-fold cross-validation**.
5. Rank compounds based on predicted longevity impact.
```

## Current Work
```markdown
- Testing different regression models, including **Chemprop** ([repo](https://github.com/chemprop/chemprop)) and **Molformer** ([repo](https://github.com/IBM/molformer)).
- Evaluating performance via **coefficient of determination (R²)**.
- Investigating lifespan effects across multiple species: *Caenorhabditis elegans, Drosophila melanogaster, Mus musculus*.
- Visualizing model performance ([example](https://seaborn.pydata.org/examples/grouped_boxplot.html)).
```

## Code Structure
```markdown
- **Data Processing:** Extracts and transforms molecular data.
- **Model Training:** Implements and evaluates regression models.
- **Results Analysis:** Compares model predictions and rankings.
```
