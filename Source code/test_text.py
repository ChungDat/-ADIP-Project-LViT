import sys
from utils import read_text
import Config as config

if config.task_name == 'MoNuSeg':
    train_text = read_text(config.train_dataset + 'Train_text.xlsx')
    val_text = read_text(config.val_dataset + 'Val_text.xlsx')
elif config.task_name == 'Covid19':
    train_text = read_text(config.train_dataset + 'Train_text.xlsx')
    val_text = read_text(config.val_dataset + 'Val_text.xlsx')

print(f"Number of train texts: {len(train_text)}")
for k, v in list(train_text.items())[:2]:
    print(f"Key: {k}, Value: {repr(v)}")
