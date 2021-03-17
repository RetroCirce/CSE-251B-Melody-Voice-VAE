import torch
from torch.utils.data import Dataset


class DynamicDataset(Dataset):
         def __init__(self, data, max_len=320, eos_token_id=1, pad_token_id=0):
                    self.data = data
                    self.max_len = max_len
                    self.eos_token_id = eos_token_id
                    self.pad_token_id = pad_token_id

         def __len__(self):
                    return len(self.data)
         def __getitem__(self, idx):
                    item = self.data[idx]
                    lens = len(item)
                    if lens != self.max_len:
                            item = item + [self.eos_token_id] + [self.pad_token_id] * (self.max_len - len(item) - 1)
                    item = {
                        'data': torch.tensor(item, dtype=torch.long),
                        'lens': torch.tensor(lens, dtype=torch.long)                                                                                         } 
                    return item
