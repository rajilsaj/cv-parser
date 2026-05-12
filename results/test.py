import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python test.py <path_to_csv>")
    sys.exit(1)

file_path = sys.argv[1]
try:
    df_train = pd.read_csv(file_path)
    if 'label' in df_train.columns:
        print(f"Value counts for 'label' in {file_path}:")
        print(df_train['label'].value_counts())
    else:
        print(f"Error: 'label' column not found in {file_path}")
        print(f"Available columns: {list(df_train.columns)}")
except Exception as e:
    print(f"Error reading {file_path}: {e}")
