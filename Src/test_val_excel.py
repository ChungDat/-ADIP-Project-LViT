import pandas as pd
import Config as config

df = pd.read_excel(config.val_dataset + 'Val_text.xlsx')
print(f"Val_text.xlsx shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
if len(df) > 0:
    print(df.head())
else:
    print("DataFrame is empty!")
