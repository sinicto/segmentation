import cv2
from torch.utils.data import DataLoader, Dataset

class Iterator(Dataset):
    def __init__(self, data_dict, batch_size=32, shuffle=False, num_workers=1):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.iter = DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __getitem__(self, index):
        return {k: self.data_dict[k][index] for k in self.old_keys}

    def __len__(self):
        return self.data_dict["img"].shape[0]
