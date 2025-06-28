import pandas as pd

def main():
    # Read the CSV file with low_memory set to False to avoid dtype warnings.
    print("Reading CSV file...")
    df = pd.read_csv("0bc5ea69-fcc7-4998-ab6c-70c3a0df778b.csv", low_memory=False)
    
    # Basic dataset insights
    print("\nDataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    print("\nDescriptive statistics for numeric columns:")
    print(df.describe())
    
    # For categorical columns with low cardinality, print their value counts.
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        if df[col].nunique() < 20:
            print(f"\nValue counts for '{col}':")
            print(df[col].value_counts())
    
    # Compute and print a correlation matrix for numeric variables.
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        corr_matrix = df[numeric_cols].corr()
        print("\nCorrelation Matrix:")
        print(corr_matrix)
    else:
        print("\nNo numeric columns found for correlation analysis.")
    
    print("\nEDA complete.")

if __name__ == '__main__':
    main()