import torch
from torch.utils.data import Dataset


class DynamicDataset(Dataset):
    def __init__(self, data, max_len=320, pad_token_id=0):
        self.data = data
        self.max_len = max_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        lens = len(item)
        item = item + [self.pad_token_id] * (self.max_len - len(item))
        item = {
            'data': torch.tensor(item, dtype=torch.long),
            'lens': torch.tensor(lens, dtype=torch.long)
        }
        return item
