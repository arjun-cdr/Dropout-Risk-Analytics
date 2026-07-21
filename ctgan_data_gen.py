import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

# 1. Create a dummy dataset (or load your real dataset with pd.read_csv)
data = pd.DataFrame({
    'age': np.random.randint(18, 70, size=500),
    'income': np.random.normal(55000, 15000, size=500).round(2),
    'education': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], size=500),
    'purchased': np.random.choice(['Yes', 'No'], size=500)
})

print("--- Real Data Sample ---")
print(data.head())

# 2. Automatically detect table metadata (column types)
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

# 3. Initialize and train CTGAN Synthesizer
# (For production/real data, set epochs to 300+ for better quality)
synthesizer = CTGANSynthesizer(
    metadata,
    epochs=100,
    verbose=True
)

print("\n--- Training CTGAN ---")
synthesizer.fit(data)

# 4. Generate synthetic data rows
synthetic_data = synthesizer.sample(num_rows=1000)

print("\n--- Synthetic Data Sample ---")
print(synthetic_data.head())

# 5. Evaluate how well synthetic data matches real data distributions
quality_report = evaluate_quality(data, synthetic_data, metadata)
