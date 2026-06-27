import pandas as pd
df = pd.read_excel('./datasets/Covid19/Test_Folder/Test_text.xlsx')
print(f"Test_text.xlsx shape: {df.shape}")
print(df.head())
