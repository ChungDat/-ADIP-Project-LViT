import pandas as pd
train_df = pd.read_excel('./datasets/Covid19/Train_Folder/Train_text.xlsx')
test_df = pd.read_excel('./datasets/Covid19/Test_Folder/Test_text.xlsx')
key = 'mask_sub-S11596_ses-E21420_run-1_bp-chest_vp-ap_dx.png'
print(f"In train: {key in train_df['Image'].values}")
print(f"In test: {key in test_df['Image'].values}")
