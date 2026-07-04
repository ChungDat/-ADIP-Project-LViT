from utils import read_text
import Config as config

val_text = read_text(config.val_dataset + 'Val_text.xlsx')
keys = list(val_text.keys())
print(f"Number of keys: {len(keys)}")
print(f"First 5 keys: {keys[:5]}")
print(f"Is 'mask_sub-S11596_ses-E21420_run-1_bp-chest_vp-ap_dx.png' in keys? {'mask_sub-S11596_ses-E21420_run-1_bp-chest_vp-ap_dx.png' in val_text}")

import os
val_masks = os.listdir(config.val_dataset + 'labelcol')
print(f"First 5 masks in folder: {val_masks[:5]}")
