import pandas as pd
df = pd.read_excel('./datasets/Covid19/Val_ID.xlsx')
print(f"Val_ID.xlsx shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
if len(df) > 0:
    print(df.head())
