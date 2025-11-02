import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

dataset_cfg = config['dataset']

def inspect_dataset():
    # Load CSV file
    df = pd.read_csv(dataset_cfg['file'])
    
    # Select only the configured features
    features = dataset_cfg['features']
    label_col = dataset_cfg.get('label_col', None)
    
    df = df[features + [label_col]] if label_col else df[features]
    
    print(f"[INFO] Loaded {len(df)} records from {dataset_cfg['file']}")
    print(f"[INFO] Features used: {features}")
    print(f"[INFO] Label column: {label_col}")
    print("\n[INFO] First five rows:")
    print(df.head())
    
    # print("\n[INFO] Missing values per column:")
    # print(df.isnull().sum())

    print("\n[INFO] Summary statistics (numeric features):")
    print(df.describe())
    
    if label_col:
        print(f"\n[INFO] Value counts for label column '{label_col}':")
        print(df[label_col].value_counts())

    # Visualizations
    # Numeric features histograms
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    df[numeric_cols].hist(figsize=(15, 8), bins=20)
    plt.suptitle("Feature Distributions")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
        plt.title("Feature Correlation Matrix")
        plt.show()

    # Categorical features countplot (if exists)
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        plt.figure()
        sns.countplot(x=col, data=df)
        plt.title(f"Value Counts for {col}")
        plt.show()
    
    return df

if __name__ == "__main__":
    inspect_dataset()