import pandas as pd
import yaml

# Load dataset configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

dataset_cfg = config['dataset']

def load_dataset():
    # Load CSV file
    df = pd.read_csv(dataset_cfg['file'])
    
    # Select only the configured features
    features = dataset_cfg['features']
    label_col = dataset_cfg.get('label_col', None)
    
    df = df[features + [label_col]] if label_col else df[features]
    
    sample_size = dataset_cfg.get('sample_size', len(df))
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print(f"[INFO] Loaded {len(df)} records from {dataset_cfg['file']}")
    print(f"[INFO] Features used: {features}")
    print(f"[INFO] Label column: {label_col}")
    print(df.head())
    
    return df

if __name__ == "__main__":
    load_dataset()
